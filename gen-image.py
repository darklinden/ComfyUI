#!/usr/bin/env python3
"""gen-image.py — 用同目录的 workflow.json 无头出图（不起 web 服务、不开浏览器）。

    ./gen-image.py "a cozy cat sleeping on a windowsill, cinematic lighting"
    ./gen-image.py "a cozy cat" --batch-count 3

直接在本进程里跑 ComfyUI 的执行器（PromptExecutor），不碰 HTTP/WebSocket。图片落在
ComfyUI 的 output/ 目录；每张图一个 seed（默认随机，也可用 --seed 固定），并把 seed
写进文件名：

    output/<filename_prefix>_seed3827194655_00001_.png

PNG 里会嵌入一份同步过 prompt / seed / 文件名前缀的 workflow，拖回 ComfyUI 网页版
直接 Queue 就能复现同一张图。

workflow.json 在网页版里调好后用 Workflow → Export 导出覆盖（网页版的 Ctrl+S 存的
是 user/default/workflows/，不是这个文件）。
"""

# comfy 相关的 import 一律留在函数内部，这不是疏漏：comfy/cli_args.py 在 import 期就
# 求值一次 `if comfy.options.args_parsing: parse_args()`，所以 enable_args_parsing() 必须
# 早于 comfy.cli_args 的首次 import，否则它会被定死成默认值，--cpu / --verbose 这类透传
# 参数会被静默丢掉。实测 comfy_execution.progress 和 hook_breaker_ac10a0 都会传递性
# import comfy.cli_args，因此它们同样不能上提到模块级。

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import sys
import uuid
from collections import deque
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WORKFLOW_PATH = os.path.join(SCRIPT_DIR, "workflow.json")
if os.name == "nt":
    VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "Scripts", "python.exe")
else:
    VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")

# UI 里带控件的输入类型（值存在 widgets_values 里）；其余类型（MODEL/CLIP/LATENT…）都是连线
WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})
# 语义探测时优先认这些输入名
PREFERRED_PROMPT_NAMES = ("prompt", "text")
SEED_INPUT_NAMES = ("seed", "noise_seed")
# 前端 LGraphEventMode：2 = NEVER（静音），4 = BYPASS（旁路）
SKIPPED_NODE_MODES = (2, 4)


class ConversionError(Exception):
    """workflow.json 里出现了转换器无法处理的东西。"""


def reexec_in_venv() -> None:
    """comfy / torch 只装在仓库自带的 .venv 里，用错解释器就重跑一次自己。"""
    if os.environ.get("_GEN_IMAGE_REEXEC") == "1" or not os.path.isfile(VENV_PYTHON):
        return
    if os.path.realpath(sys.executable) == os.path.realpath(VENV_PYTHON):
        return
    os.environ["_GEN_IMAGE_REEXEC"] = "1"
    os.execv(VENV_PYTHON, [VENV_PYTHON, os.path.realpath(__file__), *sys.argv[1:]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen-image.py",
        description="用同目录的 workflow.json 无头出图（不起 web 服务、不开浏览器）",
        epilog="未识别的参数会原样交给 ComfyUI 自己的参数解析（例如 --cpu、--verbose DEBUG）。",
    )
    parser.add_argument("prompt", help="正向提示词，注入 workflow 里的文本编码节点")
    parser.add_argument(
        "-n", "--batch-count", type=int, default=1, metavar="N",
        help="连续生成 N 张（默认 1），每张一个随机 seed",
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="SEED",
        help="固定随机种子，第 i 张用 SEED+i；不给则每张随机（默认）",
    )
    parser.add_argument(
        "--dump-prompt", action="store_true",
        help="只打印转换结果（prompt + 同步后的 workflow）不跑模型，"
             "也不施加随机 seed / 文件名前缀改写，便于和已有 PNG 做回归比对",
    )
    return parser


def parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = build_parser()
    cli, passthrough = parser.parse_known_args(argv)
    if cli.batch_count < 1:
        parser.error("--batch-count 至少为 1")
    if cli.seed is not None and cli.seed < 0:
        parser.error("--seed 不能为负")
    # comfy.cli_args 在 import 期就会读 sys.argv，把不认识的参数留给它
    sys.argv = [sys.argv[0], *passthrough]
    return cli


def bootstrap():
    """按 main.py 的顺序把 ComfyUI 拉起来，但不建 PromptServer、不启队列线程。

    顺序敏感：先把我们的参数从 sys.argv 里摘掉（parse_cli），再让 comfy 解析。
    """
    import comfy.options
    comfy.options.enable_args_parsing()  # 必须在 import comfy.cli_args 之前

    from comfy.cli_args import args
    import folder_paths
    import comfy.utils
    import comfy.model_management
    import nodes
    import execution

    if args.output_directory:
        folder_paths.set_output_directory(os.path.abspath(args.output_directory))

    return argparse.Namespace(
        args=args,
        folder_paths=folder_paths,
        comfy_utils=comfy.utils,
        model_management=comfy.model_management,
        nodes=nodes,
        execution=execution,
    )


def init_nodes(ctx) -> None:
    """加载内置节点与自定义节点（UnetLoaderGGUF 就在 custom_nodes/ 里）。"""
    import hook_breaker_ac10a0

    hook_breaker_ac10a0.save_functions()
    try:
        asyncio.run(ctx.nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
        ctx.model_management.set_cudnn_benchmark()
    finally:
        hook_breaker_ac10a0.restore_functions()


def install_progress_hooks(ctx) -> None:
    """让进度条能动，并且让 Ctrl-C 能在采样中途生效。

    comfy/utils.py 的 PROGRESS_BAR_HOOK 默认是 None，没人装它的话 tqdm 永远不动，
    而且 throw_exception_if_processing_interrupted() 不会被调用 —— 采样中按 Ctrl-C 就没反应。
    另外 execute_async 每次都会调 reset_progress_state()，它会把已注册的 handler 全清掉，
    所以要在那之后再挂 CLI 的 handler。
    """
    from comfy_execution.progress import CLIProgressHandler, get_progress_state
    from comfy_execution.utils import get_executing_context

    # 关掉 k_diffusion 自己那条 tqdm（nodes.py:1590 的 disable_pbar）：
    # 它没有节点信息，会和下面这条「Node N:」的进度条重复刷屏
    ctx.comfy_utils.set_progress_bar_enabled(False)

    class LazyCLIProgressHandler(CLIProgressHandler):
        """只给真正报进度的节点画条（加载器那种 total=1 的就不画了）。"""

        def start_handler(self, node_id, state, prompt_id):
            pass

    # comfy_execution.progress 没有暴露「注册一个跨 prompt 常驻的 handler」的公开口子，
    # reset_progress_state() 每跑一张图都会建一个空 registry，所以只能包一层，
    # 在它之后再挂 CLI handler。
    original_reset = ctx.execution.reset_progress_state

    def reset_then_add_cli_handler(prompt_id, dynprompt):
        original_reset(prompt_id, dynprompt)
        ctx.execution.add_progress_handler(LazyCLIProgressHandler())

    ctx.execution.reset_progress_state = reset_then_add_cli_handler

    def hook(value, total, preview_image=None, prompt_id=None, node_id=None):
        ctx.model_management.throw_exception_if_processing_interrupted()
        executing = get_executing_context()
        if prompt_id is None and executing is not None:
            prompt_id = executing.prompt_id
        if node_id is None and executing is not None:
            node_id = executing.node_id
        get_progress_state().update_progress(node_id, value, total, preview_image)

    ctx.comfy_utils.set_progress_bar_global_hook(hook)


def install_sigint_handler(ctx) -> None:
    pressed = []

    def handler(signum, frame):
        pressed.append(1)
        if len(pressed) == 1:
            print(  # noqa: T201
                "\n中断请求已发出，等当前节点让出控制…（再按一次 Ctrl-C 强制退出）",
                file=sys.stderr, flush=True,
            )
            ctx.nodes.interrupt_processing(True)
        else:
            print("\n强制退出", file=sys.stderr, flush=True)  # noqa: T201
            os._exit(130)

    signal.signal(signal.SIGINT, handler)


# ---------------------------------------------------------------------------
# UI 格式 workflow -> API 格式 prompt
# ---------------------------------------------------------------------------

def get_node_info(ctx, class_type: str) -> dict:
    """取节点的 v1 形状输入信息。

    server.py 里那份 node_info() 是路由闭包里的局部函数、import 不到，这里只复刻需要的
    子集（input / input_order / display_name），不追求和它逐字段对齐。

    v3 节点（io.Schema 写的，比如 TextEncodeQwenImage21）没有 INPUT_TYPES()，
    只有 GET_NODE_INFO_V1() 能给出正确形状。
    """
    from comfy_api.internal import _ComfyNodeInternal

    obj_class = ctx.nodes.NODE_CLASS_MAPPINGS[class_type]
    if issubclass(obj_class, _ComfyNodeInternal):
        info = obj_class.GET_NODE_INFO_V1()
        info.setdefault("display_name", class_type)
        return info
    input_types = obj_class.INPUT_TYPES()
    return {
        "input": input_types,
        "input_order": {key: list(value.keys()) for key, value in input_types.items()},
        "display_name": ctx.nodes.NODE_DISPLAY_NAME_MAPPINGS.get(class_type, class_type),
    }


def split_input_entry(entry):
    """INPUT_TYPES 的一项 -> (类型, 选项字典)。hidden 项是裸字符串。"""
    if isinstance(entry, (tuple, list)):
        io_type = entry[0] if entry else None
        options = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
        return io_type, options
    return entry, {}


def is_widget_input(io_type, options: dict) -> bool:
    if options.get("forceInput"):
        return False
    if isinstance(io_type, list):  # 下拉框
        return True
    return io_type in WIDGET_TYPES


def convert(ctx, workflow: dict):
    """UI workflow -> (API prompt, widget_map, warnings)。

    widget_map: node_id -> {输入名: 在 widgets_values 里的下标}，用来把改动同步回 UI 副本。
    """
    links = {}
    for link in workflow.get("links") or []:
        if isinstance(link, dict):
            links[link["id"]] = (link["origin_id"], link["origin_slot"])
        else:
            links[link[0]] = (link[1], link[2])  # [link_id, 源节点, 源槽, 目标节点, 目标槽, 类型]

    prompt = {}
    widget_map = {}
    warnings = []

    for node in workflow.get("nodes") or []:
        if node.get("mode") in SKIPPED_NODE_MODES:
            continue
        node_id = str(node["id"])
        class_type = node.get("type")
        if class_type not in ctx.nodes.NODE_CLASS_MAPPINGS:
            raise ConversionError(
                f"节点 {node_id} 的类型 {class_type!r} 没有注册（自定义节点没装好？）"
            )
        info = get_node_info(ctx, class_type)
        input_order = info["input_order"]
        sockets = {i["name"]: i for i in node.get("inputs") or []}
        widgets_values = node.get("widgets_values")
        if widgets_values is None:
            widgets_values = []
        if isinstance(widgets_values, dict):
            raise ConversionError(
                f"节点 {node_id}({class_type}) 用了具名 widgets_values，暂不支持；"
                "请在网页版里用 Workflow → Export 导出位置格式"
            )

        inputs = {}
        node_widget_map = {}
        seen = set()
        widget_index = 0

        for category in ("required", "optional"):  # hidden 由执行器自己填
            for name in input_order.get(category) or []:
                seen.add(name)
                io_type, options = split_input_entry((info["input"].get(category) or {}).get(name))
                widget = is_widget_input(io_type, options)
                socket = sockets.get(name)

                if socket is not None and socket.get("link") is not None:
                    origin = links.get(socket["link"])
                    if origin is None:
                        warnings.append(f"节点 {node_id}({class_type}) 的 {name} 指向不存在的连线 {socket['link']}")
                    else:
                        inputs[name] = [str(origin[0]), origin[1]]
                    if widget:
                        widget_index += 1  # 被连线占用的控件位在前端序列化里仍然占一格
                elif widget:
                    if widget_index < len(widgets_values):
                        inputs[name] = widgets_values[widget_index]
                        node_widget_map[name] = widget_index
                    else:
                        warnings.append(f"节点 {node_id}({class_type}) 的 widgets_values 不够，{name} 没取到值")
                    widget_index += 1

                if widget and options.get("control_after_generate"):
                    widget_index += 1  # seed 后面那个「control_after_generate」下拉框不是输入

        # autogrow 之类展开出来的动态连线：出现在节点 inputs 里但不在 input_order 里
        for name, socket in sockets.items():
            if name in seen or socket.get("link") is None:
                continue
            origin = links.get(socket["link"])
            if origin is not None:
                inputs[name] = [str(origin[0]), origin[1]]

        if widget_index != len(widgets_values):
            warnings.append(
                f"节点 {node_id}({class_type}) 按序吃掉了 {widget_index} 个 widgets_values，"
                f"实际有 {len(widgets_values)} 个 —— 位置映射可能错位"
            )

        prompt[node_id] = {
            "inputs": inputs,
            "class_type": class_type,
            "_meta": {"title": node.get("title") or info.get("display_name") or class_type},
        }
        widget_map[node_id] = node_widget_map

    if not prompt:
        raise ConversionError("workflow 里没有可执行的节点")
    return prompt, widget_map, warnings


# ---------------------------------------------------------------------------
# 语义探测：往哪个节点注入 prompt / seed / 文件名前缀
# ---------------------------------------------------------------------------

def find_sampler(prompt: dict):
    for node_id, node in prompt.items():
        if "latent_image" in node["inputs"] and "positive" in node["inputs"]:
            return node_id
    for node_id, node in prompt.items():
        if {"model", "positive", "negative"} <= set(node["inputs"]):
            return node_id
    return None


def is_link(value) -> bool:
    """连线值一定是 [str(源节点id), 源槽]（转换器自己写的），控件值不会长这样。"""
    return (
        isinstance(value, list) and len(value) == 2
        and isinstance(value[0], str) and value[0].lstrip("-").isdigit()
        and isinstance(value[1], int)
    )


def find_prompt_target(prompt: dict, widget_map: dict, sampler_id: str):
    """从采样器的 positive 逆向上溯，找那个代表正向提示词的字符串控件。

    返回 ((深度, 节点id, 输入名), 歧义候选列表)。找不到或歧义时第一项为 None。
    """
    candidates = []
    seen = set()
    queue = deque()
    positive = prompt[sampler_id]["inputs"].get("positive")
    if is_link(positive):
        queue.append((str(positive[0]), 0))

    while queue:
        node_id, depth = queue.popleft()
        if node_id in seen or node_id not in prompt:
            continue
        seen.add(node_id)
        node = prompt[node_id]
        for name in widget_map.get(node_id) or {}:
            if isinstance(node["inputs"].get(name), str):
                candidates.append((depth, node_id, name))
        for value in node["inputs"].values():
            if is_link(value):
                queue.append((str(value[0]), depth + 1))

    if not candidates:
        return None, []
    for preferred in PREFERRED_PROMPT_NAMES:
        named = sorted(c for c in candidates if c[2] == preferred)
        if named:
            nearest = [c for c in named if c[0] == named[0][0]]
            return (nearest[0], []) if len(nearest) == 1 else (None, nearest)
    if len(candidates) == 1:
        return candidates[0], []
    return None, sorted(candidates)


def find_seed_target(widget_map: dict, sampler_id: str):
    node_widget_map = widget_map.get(sampler_id) or {}
    for name in SEED_INPUT_NAMES:
        if name in node_widget_map:
            return name
    return None


def find_save_prefix_target(prompt: dict, widget_map: dict):
    """找存图节点的 filename_prefix 控件。"""
    fallback = None
    for node_id, node in prompt.items():
        if "filename_prefix" not in (widget_map.get(node_id) or {}):
            continue
        if "images" not in node["inputs"]:
            continue
        if node["class_type"] == "SaveImage":
            return node_id, "filename_prefix"
        fallback = fallback or (node_id, "filename_prefix")
    return fallback if fallback else (None, None)


def set_input(prompt: dict, ui_workflow: dict, widget_map: dict, node_id: str, name: str, value) -> None:
    """改 API prompt，同时把值同步进要嵌到 PNG 里的 UI workflow 副本。"""
    prompt[node_id]["inputs"][name] = value
    index = (widget_map.get(node_id) or {}).get(name)
    if index is None:
        return
    for node in ui_workflow.get("nodes") or []:
        if str(node["id"]) == node_id:
            values = node.setdefault("widgets_values", [])
            while len(values) <= index:
                values.append(None)
            values[index] = value
            return


def describe_candidates(prompt: dict, candidates) -> str:
    lines = []
    for depth, node_id, name in candidates:
        lines.append(f"  - 节点 {node_id} ({prompt[node_id]['class_type']}) 的 {name}  ← 深度 {depth}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

def build_executor(ctx):
    class StubServer:
        """PromptExecutor 只需要这几个成员：comfy_execution/server_protocol.py 的
        ExecutionServer Protocol 要求 client_id / last_node_id / sockets_metadata。"""

        client_id = None
        last_node_id = None
        sockets_metadata = {}

        def send_sync(self, *args, **kwargs):
            pass

        def queue_updated(self):
            pass

    return ctx.execution.PromptExecutor(
        StubServer(),
        cache_type=ctx.execution.CacheType.CLASSIC,  # 别用 NONE，那样每张图都要重载模型
        cache_args={"ram": 0, "ram_inactive": 0},   # execute_async 会无条件下标取它，不能省
    )


def collect_image_paths(ctx, executor) -> list[str]:
    history = getattr(executor, "history_result", None) or {}
    paths = []
    for node_output in (history.get("outputs") or {}).values():
        for item in node_output.get("images") or []:
            base = ctx.folder_paths.get_directory_by_type(item.get("type", "output")) \
                or ctx.folder_paths.get_output_directory()
            paths.append(os.path.join(base, item.get("subfolder") or "", item["filename"]))
    return paths


def was_interrupted(executor) -> bool:
    return any(event == "execution_interrupted" for event, _ in executor.status_messages)


def report_failure(executor) -> None:
    for event, data in executor.status_messages:
        if event == "execution_error":
            logging.error(
                "执行失败：节点 %s (%s)\n%s\n%s",
                data.get("node_id"), data.get("node_type"),
                data.get("exception_message", ""),
                "".join(data.get("traceback") or []),
            )
        elif event == "execution_interrupted":
            logging.error("执行中断于节点 %s (%s)", data.get("node_id"), data.get("node_type"))


def run_prompt(ctx, executor, api_prompt, ui_workflow) -> bool:
    """跑一张图；返回是否成功。"""
    prompt_id = uuid.uuid4().hex
    valid, error, good_outputs, node_errors = asyncio.run(
        ctx.execution.validate_prompt(prompt_id, api_prompt, None)
    )
    if not valid:
        logging.error("workflow 校验不通过：%s", json.dumps(error or node_errors, ensure_ascii=False, indent=2))
        return False

    executor.execute(
        api_prompt, prompt_id,
        extra_data={"extra_pnginfo": {"workflow": ui_workflow}},
        execute_outputs=good_outputs,
    )
    if not executor.success:
        report_failure(executor)
        return False
    for path in collect_image_paths(ctx, executor):
        print(path, flush=True)  # noqa: T201
    return True


def main() -> int:
    reexec_in_venv()
    cli = parse_cli(sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    if not os.path.isfile(WORKFLOW_PATH):
        logging.error(
            "找不到 workflow 文件：%s\n"
            "请先在 ComfyUI 网页版里调好图，用 Workflow → Export 导出覆盖它。", WORKFLOW_PATH,
        )
        return 2
    with open(WORKFLOW_PATH, encoding="utf-8") as handle:
        workflow = json.load(handle)

    ctx = bootstrap()
    install_progress_hooks(ctx)
    install_sigint_handler(ctx)
    init_nodes(ctx)

    try:
        api_prompt, widget_map, warnings = convert(ctx, deepcopy(workflow))
    except ConversionError as exc:
        logging.error("workflow 转换失败：%s", exc)
        return 2
    for warning in warnings:
        logging.warning("转换告警：%s", warning)

    sampler_id = find_sampler(api_prompt)
    if sampler_id is None:
        logging.error("没找到采样器节点（既没有 latent_image，也没有 model/positive/negative 输入）")
        return 2

    target, ambiguous = find_prompt_target(api_prompt, widget_map, sampler_id)
    if target is None:
        logging.error(
            "没定位到唯一的目标提示词输入（采样器 %s 的上游）：\n%s\n"
            "请在网页版里把提示词节点整理清楚再导出，或在 workflow.json 里确认节点结构。",
            sampler_id, describe_candidates(api_prompt, ambiguous),
        )
        return 2
    _, prompt_node, prompt_input = target

    seed_input = find_seed_target(widget_map, sampler_id)
    if seed_input is None:
        logging.warning("采样器 %s 上没有 seed 控件，这张 workflow 的 seed 不会被随机化", sampler_id)
    save_node, prefix_input = find_save_prefix_target(api_prompt, widget_map)
    if save_node is None:
        logging.warning("没找到带 filename_prefix 的存图节点，文件名不会带 seed")

    if cli.dump_prompt:
        ui_copy = deepcopy(workflow)
        set_input(api_prompt, ui_copy, widget_map, prompt_node, prompt_input, cli.prompt)
        json.dump({"prompt": api_prompt, "workflow": ui_copy}, sys.stdout, ensure_ascii=False, indent=2)
        print()  # noqa: T201
        return 0

    executor = build_executor(ctx)
    base_prefix = api_prompt[save_node]["inputs"][prefix_input] if save_node else None
    logging.info(
        "提示词 -> 节点 %s 的 %s；采样器 %s；存图节点 %s",
        prompt_node, prompt_input, sampler_id, save_node,
    )

    failures = 0
    for index in range(cli.batch_count):
        # 给了 --seed 就从它往后排（第 i 张 seed+i），并回绕到与随机路径同一区间
        seed = random.randrange(0, 2 ** 32) if cli.seed is None else (cli.seed + index) % (2 ** 32)
        ui_copy = deepcopy(workflow)
        api_copy = deepcopy(api_prompt)
        set_input(api_copy, ui_copy, widget_map, prompt_node, prompt_input, cli.prompt)
        if seed_input is not None:
            set_input(api_copy, ui_copy, widget_map, sampler_id, seed_input, seed)
        if save_node is not None:
            set_input(api_copy, ui_copy, widget_map, save_node, prefix_input, f"{base_prefix}_seed{seed}")

        if cli.batch_count > 1:
            logging.info("[%d/%d] seed=%s 开始生成…", index + 1, cli.batch_count, seed)
        else:
            logging.info("seed=%s 开始生成…", seed)

        if not run_prompt(ctx, executor, api_copy, ui_copy):
            if was_interrupted(executor):
                return 130
            failures += 1

    if failures:
        logging.error("%d/%d 张失败", failures, cli.batch_count)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
