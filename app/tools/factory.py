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
        logger.info("记忆工具集已创建")

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
        metadata: str = "{}",
    ) -> str:
        """
        将内容保存到长期记忆。

        参数:
            novel_id: 小说ID
            category: 分类 (character/setting/plot)
            name: 条目名称
            content: 内容
            metadata: JSON 格式的附加元数据
        """
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            meta = {}

        if category == "character":
            from app.models.memory import CharacterProfile
            profile = CharacterProfile(
                name=name, personality=content, background=meta.get("background", ""),
                role_type=meta.get("role_type", "supporting"),
                motivation=meta.get("motivation", ""), appearance=meta.get("appearance", ""),
            )
            char_id = self._ltm.save_character(novel_id, profile)
            return f"人物已保存: {name} (id={char_id})"
        elif category == "setting":
            from app.models.memory import WorldSettingProfile
            profile = WorldSettingProfile(
                name=name, description=content, category=meta.get("category", "general"),
                details=metadata,
            )
            sid = self._ltm.save_world_setting(novel_id, profile)
            return f"设定已保存: {name} (id={sid})"
        elif category == "plot":
            start_ch = meta.get("start_chapter", 0)
            end_ch = meta.get("end_chapter", 0)
            pid = self._ltm.save_main_plot(novel_id, name, content, start_ch, end_ch)
            return f"情节已保存: {name} (id={pid})"
        else:
            return f"未知分类: {category}"

    def _get_novel_outline(self, novel_id: str) -> str:
        """
        获取小说完整大纲。

        参数:
            novel_id: 小说ID
        """
        outlines = self._ltm.get_outlines(novel_id)
        if not outlines:
            return "暂无大纲数据"

        data = [
            {"章节": o.chapter_seq, "卷": o.volume, "标题": o.title,
             "概要": o.summary, "关键事件": o.key_events, "状态": o.status}
            for o in outlines
        ]
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    def _get_character_profile(self, novel_id: str, char_name: str = "") -> str:
        """
        获取人物档案。

        参数:
            novel_id: 小说ID
            char_name: 人物名称（留空则返回所有人物）
        """
        chars = self._ltm.get_characters(novel_id)
        if not chars:
            return "暂无人物数据"

        filtered = [c for c in chars if not char_name or c.name == char_name]
        data = [
            {"姓名": c.name, "角色": c.role_type, "性别": c.gender, "年龄": c.age,
             "外貌": c.appearance, "性格": c.personality, "背景": c.background,
             "动机": c.motivation, "能力": c.abilities}
            for c in filtered
        ]
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    def _get_world_building(self, novel_id: str) -> str:
        """
        获取世界观设定。

        参数:
            novel_id: 小说ID
        """
        settings = self._ltm.get_world_settings(novel_id)
        if not settings:
            return "暂无世界观设定数据"

        data = [
            {"分类": s.category, "名称": s.name, "描述": s.description}
            for s in settings
        ]
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    # ─── 短期记忆工具 ──────────────────────────────────

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
        logs = self._stm.get_recent_logs(f"novel_{novel_id}", limit=10)
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
                session_id=f"novel_{novel_id}", agent_name="mcp_adapter",
                action="manual_update", output_summary=content, novel_id=novel_id,
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
        d_id = self._stm.save_draft(
            novel_id=novel_id, chapter_seq=chapter_seq, content=content,
            title=title, feedback=feedback,
            quality_score=quality_score if quality_score > 0 else None,
        )
        return f"章节已保存: draft_id={d_id}"

    # ─── 向量搜索工具 ──────────────────────────────────

    def _search_similar_content(
        self, query: str, collection: str = "chapter_content", k: int = 5,
    ) -> str:
        """
        向量搜索相似内容。

        参数:
            query: 搜索查询
            collection: 搜索的 collection 名 (novel_characters/novel_settings/novel_plots/chapter_content)
            k: 返回数量
        """
        results = self._vs.search(collection, query, k=k)
        if not results:
            return f"在 {collection} 中未找到相关内容"
        lines = []
        for i, r in enumerate(results, 1):
            score_str = f"(score={r['score']:.3f})" if r.get("score") else ""
            lines.append(f"[{i}] {score_str} {r['content'][:300]}")
        return "\n\n".join(lines)

    # ─── 管理工具 ──────────────────────────────────────

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
        self, novel_id: str, issue_type: str, description: str,
        suggestion: str = "", chapter_seq: int = 0,
        found_by: str = "", severity: str = "medium",
    ) -> str:
        """
        记录一条写作历史问题到长期记忆，供撰写者后续规避。

        参数:
            novel_id: 小说ID
            issue_type: 问题类型 (连贯性/人物一致性/逻辑/文笔/世界观/节奏)
            description: 问题的具体描述
            suggestion: 如何规避该问题的建议
            chapter_seq: 发现问题的章节序号
            found_by: 发现者 (editor/reader)
            severity: 严重程度 (low/medium/high)
        """
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

        if bible.get("characters"):
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
        return [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16]
