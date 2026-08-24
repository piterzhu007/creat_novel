"""
工具工厂模块：基于闭包工厂创建记忆系统工具。

消除了全局变量模式，每个记忆后端通过闭包捕获。
"""

import json
from typing import Optional

import yaml
from langchain_core.tools import tool
from loguru import logger

from app.memory import LongTermMemory, ShortTermMemory, VectorStore


# 通用易混淆字符表（字符级，跨项目通用）：正确字 → 易错写法（同音/近形/简繁字）。
# 用于按「每个小说的权威角色名表」动态派生该小说的「正确名 → 易错变体」映射，
# 从源头杜绝正文人名混乱。多项目隔离：不同项目各自用自己的名表派生，互不污染。
# 用显式码点书写，避免简繁（风0x98CE/風0x98A8、枫0x67AB/楓0x6953）在源码里肉眼难辨。
_CONFUSABLE_CHARS = {
    "枫": ("峰", "风", "風", "楓", "栴"),  # 枫(U+67AB) ← 峰/风/風/楓/栴
    "荆": ("荊",),                          # 荆(U+8346) ← 荊(U+834A)
    "静": ("靖",),
    "晓": ("小",),
    "瑶": ("遥",),
    "桦": ("华",),
    "奕": ("亦",),
}

# 字段标签关键词：`**X**：` 粗体条目里若含这些词，说明是人物档案的字段标签
# （外貌/身世/性格/动机/行为…），不是角色名。用「包含」判断而非「精确匹配」，
# 避免「关键行为与动机」「结局走向」这类组合标签漏判成角色名。
_FIELD_LABEL_KEYWORDS = (
    "外貌", "身世", "性格", "特殊", "情感", "转变", "经历", "身份", "关系",
    "动机", "行为", "结局", "状态", "环境", "线索", "矛盾", "钩子", "时间",
)


class NovelMemoryTools:
    """
    小说记忆工具集 —— 基于闭包工厂的工具创建。

    将所有记忆系统操作封装为 langchain 工具，
    通过闭包捕获 ltm/stm/vs 实例，无需全局变量。
    """

    def __init__(
        self,
        ltm: LongTermMemory,
        stm: ShortTermMemory,
        vs: VectorStore,
    ):
        self._ltm = ltm
        self._stm = stm
        self._vs = vs
        # 多项目隔离：源文档 / 权威角色名表 / 易错字表都按 novel_id 缓存，不再全局共享。
        self._source_docs_read: set[str] = set()                     # 已读源文档的 novel_id 集合
        self._authoritative_names_by_novel: dict[str, set[str]] = {}  # novel_id -> 权威名表
        self._confusables_by_novel: dict[str, dict] = {}              # novel_id -> {正确名: 易错变体}
        logger.info("记忆工具集已创建（多项目隔离：源文档/名表/易错字表均按 novel_id 缓存）")

    # ─── 长期记忆工具 ──────────────────────────────────

    def _search_long_term_memory(
        self, query: str, novel_id: str, category: str = "all", k: int = 5
    ) -> str:
        """
        语义检索长期记忆。

        从人物、设定、情节等长期记忆中搜索与 query 相关的内容。
        参数:
            query: 搜索查询
            novel_id: 小说ID
            category: 搜索分类 (all/character/setting/plot)
            k: 返回结果数量
        """
        results = self._ltm.search_semantic(novel_id, query, category, k)
        if not results:
            return f"未在长期记忆中找到与 '{query}' 相关的内容。"
        lines = []
        for i, entry in enumerate(results, 1):
            lines.append(f"[{i}] [{entry.category}] {entry.content[:500]}")
        return "\n\n".join(lines)

    def _save_to_long_term(
        self, novel_id: str, category: str, name: str, content: str,
        metadata: str = "{}", locked: bool = False, allow_new: bool = False,
    ) -> str:
        """
        将内容保存到长期记忆。

        参数:
            novel_id: 小说ID
            category: 分类 (character/setting/plot/outline)
            name: 条目名称
            content: 内容
            metadata: JSON 格式的附加元数据
            locked: True=supervisor 定稿（可覆盖已锁定条目），False=子智能体草稿
            allow_new: True=supervisor 授权新增名表外角色（仅 category=character 生效）
        """
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            meta = {}

        # 写时纠正：content 里出现角色名易错写法（如「林峰」代替「枫」）时自动回退正确写法，
        # 让大纲/设定/人物正文从源头与权威名表一致，避免「故事圣经 vs 大纲」的人名冲突。
        content, corrections = self._correct_name_confusables(novel_id, content)
        note = ("（已自动纠正角色名：" + "；".join(corrections) + "）") if corrections else ""

        if category == "character":
            # 权威名表硬校验：角色名必须精确匹配该小说的源文档角色名表，拦截音近/错字幻觉
            authoritative = self._get_authoritative_names(novel_id)
            if authoritative and name not in authoritative and not allow_new:
                hint = "、".join(sorted(authoritative))
                return (
                    f"角色名「{name}」不在源文档角色名表里，已拒绝写入。"
                    f"权威角色名：{hint}。请逐字核对后重写；若确需新增名表外角色，"
                    f"由 supervisor 用 allow_new=true 授权。"
                )
            from app.models.memory import CharacterProfile
            profile = CharacterProfile(
                name=name, personality=content, background=meta.get("background", ""),
                role_type=meta.get("role_type", "supporting"),
                motivation=meta.get("motivation", ""), appearance=meta.get("appearance", ""),
            )
            try:
                char_id = self._ltm.save_character(novel_id, profile, locked=locked)
            except ValueError as e:
                return f"已拒绝：{e}"
            # 同步写入向量库（后台，不阻塞），供语义检索
            self._vs.add_background(
                "novel_characters",
                f"{name}（{profile.role_type}）：{content} {profile.motivation} {profile.appearance}",
                metadata={"novel_id": novel_id, "char_id": char_id, "name": name},
                doc_id=char_id,
            )
            return f"人物已保存: {name} (id={char_id}){note}"
        elif category == "setting":
            from app.models.memory import WorldSettingProfile
            profile = WorldSettingProfile(
                name=name, description=content, category=meta.get("category", "general"),
                details=metadata,
            )
            try:
                sid = self._ltm.save_world_setting(novel_id, profile, locked=locked)
            except ValueError as e:
                return f"已拒绝：{e}"
            self._vs.add_background(
                "novel_settings",
                f"{name}：{content}",
                metadata={"novel_id": novel_id, "setting_id": sid, "name": name},
                doc_id=sid,
            )
            return f"设定已保存: {name} (id={sid}){note}"
        elif category == "plot":
            start_ch = meta.get("start_chapter", 0)
            end_ch = meta.get("end_chapter", 0)
            try:
                pid = self._ltm.save_main_plot(novel_id, name, content, start_ch, end_ch, locked=locked)
            except ValueError as e:
                return f"已拒绝：{e}"
            self._vs.add_background(
                "novel_plots",
                f"{name}：{content}",
                metadata={"novel_id": novel_id, "plot_id": pid, "name": name},
                doc_id=pid,
            )
            return f"情节已保存: {name} (id={pid}){note}"
        elif category == "outline":
            from app.models.memory import OutlineEntry
            entry = OutlineEntry(
                chapter_seq=meta.get("chapter_seq", 0),
                volume=meta.get("volume", 1),
                title=name,
                summary=content,
                key_events=meta.get("key_events", ""),
                foreshadowing=meta.get("foreshadowing", ""),
            )
            try:
                oid = self._ltm.save_outline(novel_id, entry, locked=locked)
            except ValueError as e:
                return f"已拒绝：{e}"
            return f"大纲已保存: {name} (id={oid}){note}"
        else:
            return f"未知分类: {category}"

    def _lock_entry(self, novel_id: str, category: str, name: str) -> str:
        """
        定稿加锁：把草稿条目 locked=True，不改内容、不重发全文。

        supervisor 审核通过后用「加锁」定稿，只需传条目名（人物名/设定名/情节名/大纲标题），
        避免用 save_to_long_term 重发全文导致的 token 浪费。仅当需要修改内容时才用
        save_to_long_term 带 locked=true 重写。

        参数:
            novel_id: 小说ID
            category: 分类 (character/setting/plot/outline)
            name: 条目名称
        """
        n = self._ltm.lock_entry(novel_id, category, name)
        if n == 0:
            return f"未找到 {category}「{name}」，无法加锁"
        return f"已定稿加锁: {category}「{name}」（{n} 条）"

    def _get_novel_progress(self, novel_id: str) -> str:
        """
        获取小说的「极简进度卡」——只含进度、角色名表、大纲目录、未解决问题数，几 KB 内。

        supervisor 日常推进时优先用它（而非 get_novel_state 全量快照），避免把全部
        人物档案/世界观/大纲塞进上下文。只有一次性审核才用 get_novel_state。

        参数:
            novel_id: 小说ID
        """
        state = self._ltm.get_full_state(novel_id)
        if not state:
            return "未找到该小说"

        lines = [
            f"# 进度卡《{state['title']}》",
            f"状态：{state['status']}  进度：第 {state['current_chapter']} / {state['target_chapters']} 章",
        ]
        if state.get("characters"):
            names = [f"{c['name']}({c['role_type']})" for c in state["characters"]]
            lines.append(f"角色：{'、'.join(names)}")
        if state.get("outlines"):
            titles = [f"{o['chapter_seq']}.{o['title']}" for o in state["outlines"]]
            lines.append(f"大纲：{' / '.join(titles)}")
        if state.get("writing_issues"):
            lines.append(f"未解决问题：{len(state['writing_issues'])} 条（用 get_writing_issues 查详情）")
        lines.append("（完整档案用 get_novel_state 一次性审核；写某章用 get_writing_context）")
        return "\n".join(lines)

    def _get_novel_outline(self, novel_id: str, chapter_seq: int = 0) -> str:
        """
        获取小说大纲。

        - chapter_seq=0：返回「精简目录」（每章只给 章节号 + 标题 + 一句话概要，截断 60 字），
          供导航/定位，避免把全部分章详纲（可能数十万字）塞进上下文。
        - chapter_seq>0：返回该章（或覆盖该章的最近一卷）的完整详纲。
        """
        outlines = self._ltm.get_outlines(novel_id)
        if not outlines:
            return "暂无大纲数据"

        if chapter_seq > 0:
            matched = [o for o in outlines if o.chapter_seq == chapter_seq]
            if matched:
                outlines = matched
            else:
                # 大纲是分卷级（chapter_seq 是卷首章/卷序），找覆盖该章的最接近一卷
                candidates = [o for o in outlines if o.chapter_seq <= chapter_seq]
                if candidates:
                    outlines = [max(candidates, key=lambda o: o.chapter_seq)]

            data = [
                {"章节": o.chapter_seq, "卷": o.volume, "标题": o.title,
                 "概要": self._fix_name_confusables(novel_id, o.summary),
                 "关键事件": self._fix_name_confusables(novel_id, o.key_events),
                 "状态": o.status}
                for o in outlines
            ]
            return yaml.dump(data, allow_unicode=True, sort_keys=False)

        # chapter_seq=0：精简目录（不返回全量概要/关键事件）
        data = [
            {"章节": o.chapter_seq, "标题": o.title,
             "概要": self._fix_name_confusables(novel_id, (o.summary or "")[:60])}
            for o in outlines
        ]
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    def _get_character_profile(self, novel_id: str, char_name: str = "") -> str:
        """
        获取人物档案（精简版，避免 token 浪费）。

        参数:
            novel_id: 小说ID
            char_name: 人物名称（留空则返回所有人物，但每个字段截断到 120 字）

        说明：
            默认对每个字段（性格/背景/动机等）截断到 120 字，避免全量拉取
            导致 token 浪费。如需某个人物的完整档案，请用 char_name 精确指定。
        """
        chars = self._ltm.get_characters(novel_id)
        if not chars:
            return "暂无人物数据"

        # 无 char_name：只返回极简清单（姓名 + 角色类型），避免全量拉取 16 人档案浪费 token。
        # 要查某人的详细档案，必须带 char_name 指定单个角色。
        if not char_name:
            data = [{"姓名": c.name, "角色": c.role_type} for c in chars]
            return (
                yaml.dump(data, allow_unicode=True, sort_keys=False)
                + "\n（要查某人详细档案，请带 char_name 参数指定单个角色）"
            )

        filtered = [c for c in chars if c.name == char_name]
        if not filtered:
            return f"未找到角色「{char_name}」"

        def _truncate(s, n=120):
            s = s or ""
            return s if len(s) <= n else s[:n] + "…"

        data = [
            {"姓名": c.name, "角色": c.role_type, "性别": c.gender, "年龄": c.age,
             "外貌": _truncate(c.appearance, 80), "性格": _truncate(c.personality),
             "背景": _truncate(c.background), "动机": _truncate(c.motivation),
             "能力": _truncate(c.abilities, 80)}
            for c in filtered
        ]
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    def _get_world_building(self, novel_id: str) -> str:
        """
        获取世界观设定（精简版）。

        参数:
            novel_id: 小说ID
        """
        settings = self._ltm.get_world_settings(novel_id)
        if not settings:
            return "暂无世界观设定数据"

        def _truncate(s, n=200):
            s = s or ""
            return s if len(s) <= n else s[:n] + "…"

        data = [
            {"分类": s.category, "名称": s.name, "描述": _truncate(s.description)}
            for s in settings
        ]
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    def _get_writing_context(self, novel_id: str, chapter_seq: int = 1) -> str:
        """
        获取写作上下文快照：一次返回 writer 写某章所需的全部状态。

        合并「故事圣经 + 本章大纲 + 短期上下文（上一章结尾）+ 历史问题」，
        把 writer 写一章前的 4~5 次零散查询压成 1 次，减少 ReAct 轮次。

        参数:
            novel_id: 小说ID
            chapter_seq: 要写的章节序号
        """
        parts = []

        bible = self._get_story_bible(novel_id)
        parts.append(bible)

        outline = self._get_novel_outline(novel_id, chapter_seq)
        if outline and outline != "暂无大纲数据":
            parts.append(f"## 本章大纲\n{outline}")

        if chapter_seq > 1:
            ctx = self._get_short_term_context(novel_id, chapter_seq)
            if ctx and ctx != "暂无短期上下文":
                parts.append(f"## 短期上下文（上一章结尾）\n{ctx}")

        issues = self._get_writing_issues(novel_id)
        if issues and issues != "暂无历史写作问题":
            parts.append(f"## 历史问题\n{issues}")

        return "\n\n".join(parts)

    # ─── 短期记忆工具 ──────────────────────────────────

    def _get_chapter(self, novel_id: str, chapter_seq: int) -> str:
        """
        读取指定章节的完整正文（含所有版本历史）。

        这是 editor/reader 审核章节时必须使用的工具——之前缺这个工具，
        导致 editor 只能靠 search_similar_content 的截断片段或 get_short_term_context
        的空结果来"猜"正文，误判「正文缺失」。

        参数:
            novel_id: 小说ID
            chapter_seq: 章节序号
        """
        latest = self._stm.get_latest_draft(novel_id, chapter_seq)
        if latest is None:
            return f"第 {chapter_seq} 章暂无草稿"

        title = latest.title or f"第{chapter_seq}章"
        lines = [f"## {title}（第 {chapter_seq} 章，v{latest.version}，{latest.word_count} 字）"]
        if latest.feedback:
            lines.append(f"审核反馈：{latest.feedback}")
        if latest.quality_score:
            lines.append(f"质量评分：{latest.quality_score}")
        lines.append("")
        lines.append(latest.content or "")
        return "\n".join(lines)

    def _read_source_docs(self, novel_id: str) -> str:
        """
        读取某小说自己的源文档（世界观设定、小说提纲、问题与建议等，多项目隔离）。

        这是 architect 首次设计时的唯一依据。源文档已在 create_novel 时按 novel_id
        落库（不再从固定路径全局读取），这里按 novel_id 读取，不同项目互不污染。
        deepagents 内置 read_file 在 Windows 上会因 validate_path 拒绝绝对路径而失败，
        因此这里从记忆库读取，绕过该 bug。

        返回:
            该小说源文档的完整内容（带文件名标注）
        """
        # 硬限制：每个小说的源文档只读一次。之后一律从记忆库取结构化设定，
        # 否则每次重读都会把 20 万 token 塞进上下文。
        if novel_id in self._source_docs_read:
            return (
                "源文档已在此前读取并消化成结构化设定，请改用 get_story_bible / get_novel_state / "
                "get_character_profile / get_world_building 从记忆库获取，不要重复读源文档。"
            )
        self._source_docs_read.add(novel_id)

        docs = self._ltm.get_source_docs(novel_id)
        if not docs:
            return "该小说尚未保存源文档（create_novel 时需提供源目录），无法读取源文档。"

        parts = []
        for doc_name, content in docs:
            parts.append(f"【{doc_name}】\n{content}")

        # 追加权威角色名表，让 architect 一眼看到必须逐字遵从的名字
        names = self._get_authoritative_names(novel_id)
        if names:
            names_line = "、".join(sorted(names))
            parts.append(f"【权威角色名表（落库时逐字校验，禁止音近/错字替换）】\n{names_line}")

        return "\n\n" + "=" * 40 + "\n\n".join(parts)

    @staticmethod
    def _extract_character_names(text: str) -> set[str]:
        """从「人物角色与背景.txt」提取权威角色名表。

        角色名位于「## 二、主要人物画像」段落，两种形式：
        1. `### 角色名（注释）` 标题行（个体 + 群体标题）
        2. 群体段落内的 `**角色名**：` 粗体条目（如 F4 成员 赵志龙/魏大鹏/宋伟/刘松、
           315寝室成员 钟奕/李邱龙/曾国欢/曾国庆 等）

        剥离括号注释、排除群体标题与字段标签。
        """
        import re
        names: set[str] = set()
        # 定位「主要人物画像」段落（到「主要事件脉络」或文末为止）
        m = re.search(r'##\s*二、主要人物画像(.*?)(?=##\s*三、|\Z)', text, re.DOTALL)
        if not m:
            return names
        section = m.group(1)
        for line in section.splitlines():
            line = line.strip()

            # 1) `### 角色名` 标题行
            mm = re.match(r'^###\s+(.+)', line)
            if mm:
                raw = mm.group(1).strip()
                # 排除群体标题（配角群 / F4 / 成员 / 寝室），但保留个体「配角」（如 小雅（配角））
                if any(kw in raw for kw in ("配角群", "F4", "成员", "寝室")):
                    continue
                name = re.sub(r'（[^）]*）', '', raw).strip().strip('"\'')
                if name:
                    names.add(name)
                continue

            # 2) `**角色名**：` 粗体条目（群体段落内的成员名）
            mm = re.match(r'^\*\*(.+?)\*\*\s*[：:]', line)
            if mm:
                raw = mm.group(1).strip()
                if any(kw in raw for kw in _FIELD_LABEL_KEYWORDS):
                    continue  # 是字段标签（如「关键行为与动机」），不是角色名
                # 一条可能含多个名字（如「曾国欢、曾国庆」），按顿号/逗号拆分
                for part in re.split(r'[、，,]', raw):
                    part = re.sub(r'（[^）]*）', '', part).strip().strip('"\'')
                    if part:
                        names.add(part)
        return names

    def _get_authoritative_names(self, novel_id: str) -> set[str]:
        """获取某小说的权威角色名表（多项目隔离，按 novel_id 缓存）。

        优先从 registry 读（create_novel 时已落库）；registry 为空则回退到该小说
        自己保存的「人物角色与背景.txt」源文档现场提取。这样落库校验始终生效，
        且不同项目各自用各自的名表，不会串。
        """
        if novel_id in self._authoritative_names_by_novel:
            return self._authoritative_names_by_novel[novel_id]

        names = set(self._ltm.get_character_registry(novel_id))
        if not names:
            # 回退：从该小说自己的源文档提取（源文档名含「人物」和「背景」）
            for doc_name, content in self._ltm.get_source_docs(novel_id):
                if "人物" in doc_name and "背景" in doc_name:
                    names = self._extract_character_names(content)
                    break
        self._authoritative_names_by_novel[novel_id] = names
        return names

    def _get_name_confusables(self, novel_id: str) -> dict[str, tuple[str, ...]]:
        """按该小说的权威名表 + 通用易混淆字符表，派生「正确名 → 易错变体」映射。

        多项目隔离：不同项目各自用自己的名表派生，不会把 A 项目的人名纠错规则
        套到 B 项目头上（如 A 项目有「林�」，B 项目没有，则 B 项目不会纠错「林峰」）。
        """
        if novel_id in self._confusables_by_novel:
            return self._confusables_by_novel[novel_id]
        confusables: dict[str, tuple[str, ...]] = {}
        for name in self._get_authoritative_names(novel_id):
            variants = set()
            for i, ch in enumerate(name):
                for wrong in _CONFUSABLE_CHARS.get(ch, ()):
                    variants.add(name[:i] + wrong + name[i + 1:])
            if variants:
                confusables[name] = tuple(sorted(variants))
        self._confusables_by_novel[novel_id] = confusables
        return confusables

    def _delete_long_term_entry(self, novel_id: str, category: str, name: str) -> str:
        """
        删除长期记忆中的条目（消除新旧并存冲突），并同步删除向量库对应向量。

        参数:
            novel_id: 小说ID
            category: 分类 (character/setting/plot/outline)
            name: 条目名称（人物名/设定名/情节名/大纲标题）
        """
        if category == "character":
            n = self._ltm.delete_character(novel_id, name)
            if n:
                self._vs.delete_by_filter("novel_characters", {"novel_id": novel_id, "name": name})
        elif category == "setting":
            n = self._ltm.delete_world_setting(novel_id, name)
            if n:
                self._vs.delete_by_filter("novel_settings", {"novel_id": novel_id, "name": name})
        elif category == "plot":
            n = self._ltm.delete_main_plot(novel_id, name)
            if n:
                self._vs.delete_by_filter("novel_plots", {"novel_id": novel_id, "name": name})
        elif category == "outline":
            n = self._ltm.delete_outline(novel_id, name)
        else:
            return f"未知分类: {category}"
        return f"已删除 {category}「{name}」共 {n} 条"

    def _get_short_term_context(self, novel_id: str, chapter_seq: int) -> str:
        """
        获取当前章节的短期上下文，包括子情节和最近的草稿。

        参数:
            novel_id: 小说ID
            chapter_seq: 章节序号
        """
        parts = []

        # 活跃子情节
        sub_plots = self._stm.get_active_sub_plots(novel_id)
        if sub_plots:
            sp_lines = [f"- {sp['content'][:200]}" for sp in sub_plots]
            parts.append(f"## 活跃子情节\n{chr(10).join(sp_lines)}")

        # 上一章草稿（重点提供「结尾」，用于情节无缝衔接）
        if chapter_seq > 1:
            prev = self._stm.get_latest_draft(novel_id, chapter_seq - 1)
            if prev:
                prev_text = prev.content or ""
                # 结尾是衔接的关键：取最后 800 字，让 writer 知道上一章停在何处
                prev_ending = prev_text[-800:] if len(prev_text) > 800 else prev_text
                prev_opening = prev_text[:200] if len(prev_text) > 200 else prev_text
                parts.append(
                    f"## 上一章（第 {chapter_seq - 1} 章）内容回顾\n"
                    f"【开头】{prev_opening}\n"
                    f"……\n"
                    f"【结尾】{prev_ending}"
                )

        # 最新操作日志
        logs = self._stm.get_recent_logs_by_novel(novel_id, limit=10)
        if logs:
            log_lines = [
                f"- [{l['agent']}] {l['action']}: {l['output_summary'][:100]}"
                for l in logs
            ]
            parts.append(f"## 最近操作\n{chr(10).join(log_lines)}")

        return "\n\n".join(parts) if parts else "暂无短期上下文"

    def _update_short_term(
        self, novel_id: str, category: str, content: str, chapter_seq: int = 0,
    ) -> str:
        """
        更新短期记忆。

        参数:
            novel_id: 小说ID
            category: 分类 (sub_plot/draft/log)
            content: 内容
            chapter_seq: 章节序号
        """
        if category == "sub_plot":
            sp_id = self._stm.add_sub_plot(novel_id, f"ch_{chapter_seq}", content)
            return f"子情节已记录: {sp_id}"
        elif category == "draft":
            d_id = self._stm.save_draft(novel_id, chapter_seq, content)
            return f"草稿已保存: {d_id}"
        elif category == "log":
            self._stm.log_agent_action(
                agent_name="mcp_adapter", action="manual_update",
                output_summary=content, novel_id=novel_id,
            )
            return "日志已记录"
        return f"未知分类: {category}"

    def _save_chapter(
        self, novel_id: str, chapter_seq: int, content: str, title: str = "",
        feedback: str = "", quality_score: float = 0.0,
    ) -> str:
        """
        保存完成的章节到短期记忆。

        参数:
            novel_id: 小说ID
            chapter_seq: 章节序号
            content: 章节正文
            title: 章节标题
            feedback: 审核反馈
            quality_score: 质量评分
        """
        # 写时硬校验：正文出现角色名易错写法（如「林峰/林风/林楓/林栴」代替「枫」）时，
        # 自动回退为正确写法后再落库——不是告警，是直接纠正，从源头杜绝正文人名混乱。
        corrected, corrections = self._correct_name_confusables(novel_id, content)

        d_id = self._stm.save_draft(
            novel_id=novel_id, chapter_seq=chapter_seq, content=corrected,
            title=title, feedback=feedback,
            quality_score=quality_score if quality_score > 0 else None,
        )
        # 同步写入向量库（用纠正后的正文），供语义检索「前面有没有类似情节/描写」
        self._vs.add_background(
            "chapter_content",
            corrected,
            metadata={"novel_id": novel_id, "chapter_seq": chapter_seq, "title": title},
            doc_id=d_id,
        )
        if corrections:
            return (
                f"章节已保存: draft_id={d_id}（已自动纠正角色名 {len(corrections)} 处："
                + "；".join(corrections) + "）"
            )
        return f"章节已保存: draft_id={d_id}"

    def _correct_name_confusables(self, novel_id: str, content: str) -> tuple:
        """纠正正文里的易混淆角色名错误写法，返回 (纠正后正文, 纠正记录列表)。

        按该小说的权威名表派生易错变体（多项目隔离），只纠正「本小说真实存在角色」的错字。
        """
        corrected = content
        corrections = []
        for correct_name, variants in self._get_name_confusables(novel_id).items():
            for v in variants:
                if v in corrected:
                    n = corrected.count(v)
                    corrected = corrected.replace(v, correct_name)
                    corrections.append(f"「{v}」→「{correct_name}」×{n}")
        return corrected, corrections

    def _fix_name_confusables(self, novel_id: str, text: str) -> str:
        """读时纠正：把文本里的易混淆角色名错误写法回退为正确写法。

        用于「读大纲/读设定」时自愈旧数据或已 locked 定稿数据里的错字，
        让读者（writer/editor）拿到的永远是权威名表里的正确写法。
        """
        if not text:
            return text
        corrected, _ = self._correct_name_confusables(novel_id, text)
        return corrected

    def _patch_chapter(self, novel_id: str, chapter_seq: int, old_text: str,
                       new_text: str = "", replace_all: bool = True) -> str:
        """
        精准替换章节正文中的文本（不返回全文，省 token）。

        用于 supervisor 直接修正小问题（错字、角色名用字、单句口径），
        比 get_chapter 读全文 + save_chapter 存全文省大量 token。

        参数:
            novel_id: 小说ID
            chapter_seq: 章节序号
            old_text: 要替换的原文（精确匹配）
            new_text: 替换后的文本（空串则删除）
            replace_all: 是否替换全部匹配（默认 True）
        """
        result = self._stm.patch_chapter(novel_id, chapter_seq, old_text, new_text, replace_all)
        n = result["found"]
        if n == 0:
            return f"第 {chapter_seq} 章中未找到「{old_text}」，未做修改"

        # 同步向量库：patch 会生成新版本草稿，旧版本的向量已过期。
        # 删除该章全部旧向量，用最新正文重建，保证 search_similar_content 只命中最新正文。
        latest = self._stm.get_latest_draft(novel_id, chapter_seq)
        if latest:
            self._vs.delete_by_filter(
                "chapter_content",
                {"novel_id": novel_id, "chapter_seq": chapter_seq},
            )
            self._vs.add_background(
                "chapter_content",
                latest.content or "",
                metadata={"novel_id": novel_id, "chapter_seq": chapter_seq,
                          "title": latest.title or ""},
                doc_id=latest.draft_id,
            )
        return f"第 {chapter_seq} 章已替换「{old_text}」→「{new_text}」共 {n} 处"

    # ─── 向量搜索工具 ──────────────────────────────────

    def _search_similar_content(
        self, query: str, novel_id: str = "", collection: str = "chapter_content", k: int = 5,
    ) -> str:
        """
        向量搜索相似内容（语义检索，RAG 记忆）。

        参数:
            query: 搜索查询
            novel_id: 小说ID（留空则跨小说搜索；传了则只在该小说内搜索，避免串数据）
            collection: 搜索的 collection 名 (novel_characters/novel_settings/novel_plots/chapter_content)
            k: 返回数量
        """
        where = {"novel_id": novel_id} if novel_id else None
        results = self._vs.search(collection, query, k=k, where=where)
        if not results:
            scope = f"（小说 {novel_id} 内）" if novel_id else ""
            return f"在 {collection} {scope}中未找到相关内容"
        lines = []
        for i, r in enumerate(results, 1):
            score_str = f"(score={r['score']:.3f})" if r.get("score") else ""
            lines.append(f"[{i}] {score_str} {r['content'][:300]}")
        return "\n\n".join(lines)

    # ─── 管理工具 ──────────────────────────────────────

    def _create_novel(self, title: str, genre: str = "", synopsis: str = "",
                      target_chapters: int = 0, source_dir: str = "") -> str:
        """
        创建或复用小说项目，返回 novel_id（多项目隔离）。

        若同名小说已存在，则复用其 novel_id（已提炼的人物/世界观/大纲仍在记忆库，
        可用 get_novel_state 接续，不必重新设计）。

        参数:
            title: 小说标题
            genre: 小说类型/流派（如玄幻、都市、科幻）
            synopsis: 小说简介
            target_chapters: 目标章节数
            source_dir: 该小说自己的源文档目录（留空则用项目根目录下的 4 个源文档）。
                        开启新项目时传新目录，源文档/权威名表按 novel_id 落库，互不污染。
        """
        existing = [n for n in self._ltm.list_novels() if n.title == title]
        if existing:
            novel_id = existing[0].novel_id
            return (
                f"小说「{title}」已存在，复用已有项目 (novel_id={novel_id})。"
                f"已提炼的人物/世界观/大纲可用 get_novel_state 一次性接续，"
                f"不要重新设计，也不要重读源文档。"
            )
        novel_id = self._ltm.create_novel(title, genre, synopsis, target_chapters)
        # 读取该小说自己的源文档并按 novel_id 落库，同时派生权威名表写进 registry（多项目隔离）
        self._ingest_source_docs(novel_id, source_dir)
        return f"小说已创建: {title} (novel_id={novel_id})"

    def _ingest_source_docs(self, novel_id: str, source_dir: str = "") -> None:
        """把源目录里的源文档按 novel_id 落库，并派生权威名表写进 registry。

        多项目隔离的关键：源文档不再从固定路径全局读取，而是每个小说创建时
        把「它自己目录下」的 4 个源文档存进 source_docs 表，再据此派生它自己的
        权威角色名表。
        """
        from pathlib import Path
        from app.core.config import PROJECT_ROOT

        base = Path(source_dir) if source_dir else PROJECT_ROOT
        doc_files = [
            "人物角色与背景.txt",
            "世界观与五行战力系统设定.txt",
            "小说提纲.txt",
            "问题与建议.txt",
        ]
        for fname in doc_files:
            p = base / fname
            if p.exists():
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    self._ltm.save_source_doc(novel_id, fname, content)
                except Exception as e:
                    logger.warning(f"源文档落库失败 {fname}: {e}")

        # 从该小说自己的「人物角色与背景.txt」派生权威名表
        names: set[str] = set()
        for doc_name, content in self._ltm.get_source_docs(novel_id):
            if "人物" in doc_name and "背景" in doc_name:
                names = self._extract_character_names(content)
                break
        if names:
            self._ltm.save_character_registry(novel_id, sorted(names))
            self._authoritative_names_by_novel[novel_id] = names

    def _list_novels(self) -> str:
        """列出所有已创建的小说项目"""
        novels = self._ltm.list_novels()
        if not novels:
            return "暂无小说项目"

        data = [
            {"ID": n.novel_id, "标题": n.title, "类型": n.genre,
             "状态": n.status, "目标章节": n.target_chapters,
             "更新时间": str(n.updated_at)}
            for n in novels
        ]
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    # ─── 写作问题库工具 ────────────────────────────────

    def _save_writing_issue(
        self, novel_id: str, issue_type: str, description: str = "",
        suggestion: str = "", chapter_seq: int = 0,
        found_by: str = "", severity: str = "medium",
    ) -> str:
        """
        记录一条写作历史问题到长期记忆，供撰写者后续规避。

        参数:
            novel_id: 小说ID
            issue_type: 问题类型 (连贯性/人物一致性/逻辑/文笔/世界观/节奏)
            description: 问题的具体描述（若为空则用 issue_type 兜底，避免校验失败）
            suggestion: 如何规避该问题的建议
            chapter_seq: 发现问题的章节序号
            found_by: 发现者 (editor/reader)
            severity: 严重程度 (low/medium/high)
        """
        # 防御：description 为空时用 issue_type 兜底，避免 pydantic 校验报 Field required
        description = (description or "").strip() or issue_type
        issue_id = self._ltm.save_writing_issue(
            novel_id=novel_id, issue_type=issue_type, description=description,
            suggestion=suggestion, chapter_seq=chapter_seq,
            found_by=found_by, severity=severity,
        )
        return f"写作问题已记录: [{issue_type}] {description[:50]}"

    def _get_writing_issues(self, novel_id: str, status: str = "open",
                            limit: int = 30) -> str:
        """
        获取该小说的历史写作问题列表，撰写前应阅读以主动规避。

        参数:
            novel_id: 小说ID
            status: 问题状态 (open/resolved，默认 open 只返回未解决的)
            limit: 返回数量上限
        """
        issues = self._ltm.get_writing_issues(novel_id, status=status, limit=limit)
        if not issues:
            return "暂无历史写作问题"

        data = [
            {"类型": i["issue_type"], "章节": i["chapter_seq"],
             "问题": i["description"], "规避建议": i["suggestion"],
             "发现者": i["found_by"], "严重度": i["severity"]}
            for i in issues
        ]
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    # ─── 章节导出工具 ──────────────────────────────────

    def _export_chapters(self, novel_id: str, output_dir: str = "") -> str:
        """
        将小说的所有章节正文导出为 .txt 文件（每章一个文件，取最新版本）。

        参数:
            novel_id: 小说ID
            output_dir: 输出目录（留空则默认导出到项目根目录下的 output 文件夹）

        返回:
            导出结果摘要
        """
        exported = self._stm.export_chapters(novel_id, output_dir=output_dir)
        if not exported:
            return f"小说 {novel_id} 暂无已保存的章节草稿"

        lines = [f"已导出 {len(exported)} 章到 {output_dir or 'output/'}:"]
        for e in exported:
            lines.append(f"  第{e['chapter_seq']}章《{e['title']}》({e['word_count']}字)")
        return "\n".join(lines)

    # ─── 故事圣经工具 ──────────────────────────────────

    def _get_story_bible(self, novel_id: str) -> str:
        """
        获取小说的「故事圣经」——精简的权威设定卡。

        这是创作前必读的锚点，用于保证角色名、身份、世界观规则的绝对统一，
        避免记忆混乱导致角色名不统一等问题。

        参数:
            novel_id: 小说ID

        返回:
            精简设定卡（角色名+身份、世界观一句话、主线对齐）
        """
        bible = self._ltm.get_story_bible(novel_id)
        if not bible:
            return "未找到该小说，请先创建小说项目"

        lines = []
        lines.append(f"# 《{bible['title']}》故事圣经")
        if bible.get("genre"):
            lines.append(f"类型：{bible['genre']}")
        if bible.get("synopsis"):
            lines.append(f"简介：{bible['synopsis']}")

        # 权威角色名表：优先读 DB 持久化的 registry（单一事实源，来自源文档），
        # 而非可能被污染/遗漏的角色档案表。
        registry = self._ltm.get_character_registry(novel_id)
        if registry:
            lines.append("\n## 角色名表（权威，禁止改名/简繁转换）")
            for name in registry:
                lines.append(f"- {name}")
        elif bible.get("characters"):
            lines.append("\n## 角色名表（权威，禁止改名）")
            for c in bible["characters"]:
                lines.append(f"- {c}")

        if bible.get("world_settings"):
            lines.append("\n## 世界观核心")
            for s in bible["world_settings"]:
                lines.append(f"- {s}")

        if bible.get("outline_summary"):
            lines.append(f"\n## 主线进度\n{bible['outline_summary']}")

        return "\n".join(lines)

    # ─── 全局状态中枢工具 ──────────────────────────────

    def _get_novel_state(self, novel_id: str) -> str:
        """
        获取小说的「全局状态中枢」完整快照。

        一次性返回世界观、人物、大纲、当前进度、校验历史等所有
        与生成任务相关的数据，供任何智能体读取全局共享状态。

        参数:
            novel_id: 小说ID
        """
        state = self._ltm.get_full_state(novel_id)
        if not state:
            return "未找到该小说"

        lines = [
            f"# 小说全局状态：《{state['title']}》",
            f"类型：{state['genre']}",
            f"状态：{state['status']}",
            f"进度：第 {state['current_chapter']} / {state['target_chapters']} 章",
        ]

        if state.get("world_settings"):
            lines.append(f"\n## 世界观设定（{len(state['world_settings'])} 条）")
            for s in state["world_settings"]:
                lines.append(f"- {s['name']}：{s['description'][:60]}")

        if state.get("characters"):
            lines.append(f"\n## 人物档案（{len(state['characters'])} 个）")
            for c in state["characters"]:
                lines.append(f"- {c['name']}（{c['role_type']}）：{c['personality'][:40]}")

        if state.get("outlines"):
            lines.append(f"\n## 大纲目录（{len(state['outlines'])} 章）")
            for o in state["outlines"]:
                lines.append(f"- 第{o['chapter_seq']}章《{o['title']}》")

        if state.get("writing_issues"):
            lines.append(f"\n## 校验历史（{len(state['writing_issues'])} 条未解决问题）")
            for i in state["writing_issues"][:5]:
                lines.append(f"- [{i['issue_type']}] {i['description'][:50]}")

        return "\n".join(lines)

    def _update_novel_progress(self, novel_id: str, current_chapter: int) -> str:
        """
        更新小说的生成进度（当前章节序号）。

        参数:
            novel_id: 小说ID
            current_chapter: 当前章节序号
        """
        self._ltm.update_novel_progress(novel_id, current_chapter)
        return f"进度已更新：第 {current_chapter} 章"

    # ─── 运行日志工具 ──────────────────────────────────

    def _read_recent_app_logs(self, limit: int = 200) -> list[str]:
        """
        读取应用运行日志（loguru 写入 data/logs/ 的日志文件）最近的若干行。

        用于把运行日志导出到专门的文件夹，方便用户追踪进度。
        """
        from pathlib import Path
        from app.core.config import PROJECT_ROOT

        log_dir = PROJECT_ROOT / "data" / "logs"
        if not log_dir.exists():
            return []

        # 取最新修改的日志文件（loguru 按天滚动命名 app_YYYY-MM-DD.log）
        log_files = sorted(
            log_dir.glob("app_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not log_files:
            return []

        try:
            text = log_files[0].read_text(encoding="utf-8")
        except Exception:
            return []

        lines = text.splitlines()
        return lines[-limit:] if len(lines) > limit else lines

    def _save_run_log(
        self, novel_id: str, stage: str, summary: str, detail: str = "",
    ) -> str:
        """
        将本次阶段性任务的运行日志以 txt 形式保存到专门的文件夹。

        每次产出结果后调用一次，把「本次产出摘要 + 应用运行日志」落盘，
        方便用户随时查看进度、审计各阶段做了什么。

        参数:
            novel_id: 小说ID
            stage: 阶段名称（如「人物设定」「第1章撰写」「第1章审核」）
            summary: 本次产出的结果摘要
            detail: 详细产出内容（可选，如完整正文、审核报告等）

        返回:
            保存的文件路径
        """
        from datetime import datetime
        from pathlib import Path
        from app.core.config import PROJECT_ROOT

        # 运行日志专用目录：output/run_logs/<novel_id>/
        log_dir = PROJECT_ROOT / "output" / "run_logs" / novel_id
        log_dir.mkdir(parents=True, exist_ok=True)

        # 收集应用运行日志尾部若干行
        app_log_lines = self._read_recent_app_logs(limit=200)

        # 收集运行追踪器记录的技术细节（token 消耗 / 工具调用）。
        # 注意：save_run_log 运行在 MCP 子进程，读不到主进程的 tracer 单例，
        # 因此改为主进程 tracer 落盘的跨进程快照文件（见 app/core/tracing.py）。
        try:
            from app.core.tracing import read_tracing_snapshot
            tracing_text = read_tracing_snapshot()
        except Exception:
            tracing_text = ""

        # 清理阶段名中的非法文件名字符
        safe_stage = stage.strip()
        for ch in r'\/:*?"<>|':
            safe_stage = safe_stage.replace(ch, "_")
        if not safe_stage:
            safe_stage = "阶段产出"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_stage}.txt"
        filepath = log_dir / filename

        lines = [
            f"# 运行日志 · {stage}",
            f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"小说ID：{novel_id}",
            "",
            "## 本次产出摘要",
            summary,
        ]
        if detail:
            lines.extend(["", "## 本次产出详情", detail])
        if tracing_text:
            lines.extend(["", tracing_text])
        if app_log_lines:
            lines.extend(["", "## 应用运行日志（最近 200 行）", *app_log_lines])

        try:
            filepath.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            return f"运行日志保存失败: {e}"

        logger.info(f"运行日志已保存: {filepath}")
        return f"运行日志已保存: {filepath}"

    # ─── 工具列表导出 ──────────────────────────────────

    def get_tools(self) -> list:
        """获取所有工具（用于传递给 LangGraph 工作流的节点）"""
        t1 = tool(self._search_long_term_memory)
        t1.name = "search_long_term_memory"
        t2 = tool(self._save_to_long_term)
        t2.name = "save_to_long_term"
        t3 = tool(self._get_novel_outline)
        t3.name = "get_novel_outline"
        t4 = tool(self._get_character_profile)
        t4.name = "get_character_profile"
        t5 = tool(self._get_world_building)
        t5.name = "get_world_building"
        t6 = tool(self._get_short_term_context)
        t6.name = "get_short_term_context"
        t7 = tool(self._update_short_term)
        t7.name = "update_short_term"
        t8 = tool(self._save_chapter)
        t8.name = "save_chapter"
        t9 = tool(self._search_similar_content)
        t9.name = "search_similar_content"
        t10 = tool(self._list_novels)
        t10.name = "list_novels"
        t11 = tool(self._save_writing_issue)
        t11.name = "save_writing_issue"
        t12 = tool(self._get_writing_issues)
        t12.name = "get_writing_issues"
        t13 = tool(self._export_chapters)
        t13.name = "export_chapters"
        t14 = tool(self._get_story_bible)
        t14.name = "get_story_bible"
        t15 = tool(self._get_novel_state)
        t15.name = "get_novel_state"
        t16 = tool(self._update_novel_progress)
        t16.name = "update_novel_progress"
        t17 = tool(self._create_novel)
        t17.name = "create_novel"
        t18 = tool(self._save_run_log)
        t18.name = "save_run_log"
        t19 = tool(self._get_chapter)
        t19.name = "get_chapter"
        t20 = tool(self._read_source_docs)
        t20.name = "read_source_docs"
        t21 = tool(self._delete_long_term_entry)
        t21.name = "delete_long_term_entry"
        t22 = tool(self._patch_chapter)
        t22.name = "patch_chapter"
        t23 = tool(self._get_writing_context)
        t23.name = "get_writing_context"
        t24 = tool(self._lock_entry)
        t24.name = "lock_entry"
        t25 = tool(self._get_novel_progress)
        t25.name = "get_novel_progress"
        return [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25]
