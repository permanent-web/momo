import streamlit as st
import pandas as pd
import os
import json
import uuid
import requests
import time
from datetime import datetime

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="学生个人模考学情查询&智能助手系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"  # 展开侧边栏显示API配置
)

# ==================== 数据加载 ====================
@st.cache_data
def load_data():
    """加载本地CSV数据文件，并将成绩相关列转换为数值类型。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Sheet_20260614.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # 将成绩列统一转为数值类型
    exam_phases = ["一模", "二模", "三模"]
    subjects = ["语文", "数学", "英语", "物理", "化学", "生物"]
    numeric_cols = []
    for phase in exam_phases:
        for subj in subjects:
            numeric_cols.append(f"{phase}{subj}")
        numeric_cols.extend([f"{phase}总分", f"{phase}班排", f"{phase}级排"])
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

df = load_data()

# ==================== DeepSeek API配置 ====================
def load_env_variables():
    """加载环境变量配置"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env_vars[key] = value
    return env_vars

env_vars = load_env_variables()
DEFAULT_API_KEY = env_vars.get("DEEPSEEK_API_KEY", "")
DEFAULT_BASE_URL = env_vars.get("LLM_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = env_vars.get("LLM_MODEL", "deepseek-chat")

def build_student_context(student_data):
    """构建学生数据上下文"""
    subjects = ["语文", "数学", "英语", "物理", "化学", "生物"]
    exams_list = ["一模", "二模", "三模"]
    
    context = f"""你是一位专业的教育数据分析助手。以下是学生{student_data['姓名']}（学号：{student_data['学号']}，班级：{student_data['班级']}）的三次模考数据：
    
【一模成绩】
"""
    for subj in subjects:
        context += f"- {subj}：{student_data[f'一模{subj}']}分\n"
    context += f"- 总分：{student_data['一模总分']}分，班级排名：{student_data['一模班排']}，年级排名：{student_data['一模级排']}\n\n【二模成绩】\n"
    
    for subj in subjects:
        context += f"- {subj}：{student_data[f'二模{subj}']}分\n"
    context += f"- 总分：{student_data['二模总分']}分，班级排名：{student_data['二模班排']}，年级排名：{student_data['二模级排']}\n\n【三模成绩】\n"
    
    for subj in subjects:
        context += f"- {subj}：{student_data[f'三模{subj}']}分\n"
    context += f"- 总分：{student_data['三模总分']}分，班级排名：{student_data['三模班排']}，年级排名：{student_data['三模级排']}\n\n"
    
    context += """请基于以上数据，用中文回答学生的问题。回答要详细、专业、有针对性，包括具体数据和可操作的建议。"""
    
    return context

def call_deepseek_api_stream(prompt, student_data, api_key, base_url, model_name):
    """调用DeepSeek API生成智能回复（流式输出）"""
    if not api_key or not base_url or not model_name:
        return None
    
    # 构建系统提示词
    system_prompt = build_student_context(student_data)
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "stream": True
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            error_msg = response.text[:500] if len(response.text) > 500 else response.text
            st.error(f"❌ API调用失败：{response.status_code} - {error_msg}")
            yield None
            return
        
        # 逐块生成响应
        for chunk in response.iter_lines():
            if chunk:
                chunk = chunk.decode('utf-8')
                # 处理 SSE 格式
                if chunk.startswith('data: '):
                    chunk = chunk[5:]
                    if chunk == '[DONE]':
                        break
                    try:
                        import json as json_lib
                        data = json_lib.loads(chunk)
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json_lib.JSONDecodeError:
                        continue
                    
    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if "api_key" in error_message.lower() or "unauthorized" in error_message.lower():
            st.error("❌ API Key 无效或未授权，请检查您的 API Key 是否正确")
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            st.error("❌ 网络连接失败，请检查网络设置或 Base URL 是否正确")
        elif "timeout" in error_message.lower():
            st.error("❌ 请求超时，请稍后重试")
        else:
            st.error(f"❌ 发生错误: {error_message}")
        yield None

# ==================== 会话状态初始化 ====================
# 当前角色（学生/老师）
if "current_role" not in st.session_state:
    st.session_state.current_role = None
# 当前选中的学生信息
if "selected_student_info" not in st.session_state:
    st.session_state.selected_student_info = None
# 当前老师选择的科目
if "teacher_subject" not in st.session_state:
    st.session_state.teacher_subject = None
# 当前老师选择的考试场次
if "teacher_exam" not in st.session_state:
    st.session_state.teacher_exam = None
# 是否为班主任视角
if "is_head_teacher" not in st.session_state:
    st.session_state.is_head_teacher = False
# 当前对话ID（用于区分不同对话）
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
# 所有对话历史记录（按学生ID存储）
if "all_chat_history" not in st.session_state:
    st.session_state.all_chat_history = {}
# 当前对话的消息列表
if "current_messages" not in st.session_state:
    st.session_state.current_messages = []
# 用户输入的API密钥（临时存储，不持久化）
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""
# 用户输入的Base URL
if "user_base_url" not in st.session_state:
    st.session_state.user_base_url = DEFAULT_BASE_URL
# 用户输入的模型名称
if "user_model" not in st.session_state:
    st.session_state.user_model = DEFAULT_MODEL

# ==================== 侧边栏API配置 ====================
def render_api_config_sidebar():
    """渲染侧边栏API配置"""
    with st.sidebar:
        st.header("⚙️ API配置")
        st.markdown("---")
        
        # API状态显示
        has_local_key = bool(DEFAULT_API_KEY)
        has_user_key = bool(st.session_state.user_api_key)
        
        if has_local_key:
            st.success("📁 已加载本地.env配置")
        elif has_user_key:
            st.success("🔐 已配置用户API Key")
        else:
            st.warning("⚠️ 未配置API密钥")
        
        st.markdown("---")
        
        # API Key输入
        st.session_state.user_api_key = st.text_input(
            "🔑 API Key", 
            type="password", 
            value=st.session_state.user_api_key,
            placeholder="输入您的DeepSeek API Key",
            help="从DeepSeek官网获取API Key"
        )
        
        # Base URL输入
        st.session_state.user_base_url = st.text_input(
            "🌐 Base URL",
            value=st.session_state.user_base_url,
            placeholder="https://api.deepseek.com",
            help="API服务器地址"
        )
        
        # 模型名称输入
        st.session_state.user_model = st.text_input(
            "🤖 模型名称",
            value=st.session_state.user_model,
            placeholder="deepseek-chat",
            help="使用的模型名称"
        )
        
        st.markdown("---")
        
        # API状态总结
        effective_key = st.session_state.user_api_key if st.session_state.user_api_key else DEFAULT_API_KEY
        if effective_key:
            st.success("✅ API已就绪，将调用AI助手")
        else:
            st.error("❌ 无有效API密钥，将使用规则引擎")
        
        # 获取有效配置
        def get_api_config():
            """获取当前有效的API配置"""
            return {
                "api_key": st.session_state.user_api_key if st.session_state.user_api_key else DEFAULT_API_KEY,
                "base_url": st.session_state.user_base_url if st.session_state.user_base_url else DEFAULT_BASE_URL,
                "model": st.session_state.user_model if st.session_state.user_model else DEFAULT_MODEL
            }
        
        return get_api_config

# 渲染侧边栏
get_api_config = render_api_config_sidebar()

# ==================== 辅助函数 ====================
def get_chat_history_file(student_id):
    """获取指定学生的对话历史文件路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    history_dir = os.path.join(base_dir, "chat_history")
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    return os.path.join(history_dir, f"{student_id}.json")

def load_chat_history(student_id):
    """加载指定学生的所有对话历史"""
    file_path = get_chat_history_file(student_id)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chat_history(student_id, history):
    """保存指定学生的对话历史"""
    file_path = get_chat_history_file(student_id)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def create_new_chat(student_id):
    """创建新对话"""
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_chat_id = chat_id
    st.session_state.current_messages = []
    # 加载历史并添加新对话记录
    history = load_chat_history(student_id)
    history[chat_id] = {
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }
    save_chat_history(student_id, history)
    st.session_state.all_chat_history = history

def add_message_to_chat(student_id, role, content):
    """添加消息到当前对话"""
    chat_id = st.session_state.current_chat_id
    if chat_id:
        history = load_chat_history(student_id)
        if chat_id in history:
            history[chat_id]["messages"].append({
                "role": role,
                "content": content,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            save_chat_history(student_id, history)
            st.session_state.all_chat_history = history
        st.session_state.current_messages.append({"role": role, "content": content})

def load_existing_chat(student_id, chat_id):
    """加载已有的对话"""
    history = load_chat_history(student_id)
    if chat_id in history:
        st.session_state.current_chat_id = chat_id
        st.session_state.current_messages = history[chat_id]["messages"]
        st.session_state.all_chat_history = history

# ==================== 入口页面 ====================
def show_entry_page():
    """展示入口选择页面"""
    st.title("🎓 模考学情查询系统")
    st.markdown(
        """
        <div style="margin-bottom: 30px; color: #555; font-size: 16px;">
            本系统为学生和老师提供模考数据查询与智能分析服务。<br>
            请选择您的身份进入相应查询页面。
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 角色选择
    st.markdown("---")
    st.subheader("👤 请选择您的身份")
    
    role_col1, role_col2 = st.columns(2)
    with role_col1:
        if st.button("👨‍🎓 学生查询", use_container_width=True, type="primary"):
            st.session_state.current_role = "student"
            st.rerun()
    with role_col2:
        if st.button("👨‍🏫 老师查询", use_container_width=True, type="primary"):
            st.session_state.current_role = "teacher"
            st.rerun()

# ==================== 学生查询入口页面 ====================
def show_student_search_page():
    """展示学生查询入口页面"""
    st.title("👨‍🎓 学生个人模考学情查询")
    st.markdown(
        """
        <div style="margin-bottom: 20px; color: #555;">
            请通过下拉选择或输入学号/姓名查询您的模考数据。
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 返回按钮
    if st.button("🔙 返回首页", key="back_to_entry"):
        st.session_state.current_role = None
        st.session_state.selected_student_info = None
        st.rerun()
    
    st.markdown("---")
    
    # 方式一：下拉选择
    st.subheader("📋 方式一：下拉选择学生")
    student_names = sorted(df["姓名"].unique())
    selected_from_dropdown = st.selectbox("选择学生姓名", student_names, key="dropdown_student")
    
    if st.button("✅ 确认选择", key="confirm_dropdown"):
        student_row = df[df["姓名"] == selected_from_dropdown].iloc[0]
        st.session_state.selected_student_info = student_row
        # 创建新对话
        create_new_chat(student_row["学号"])
        st.rerun()
    
    st.markdown("---")
    
    # 方式二：输入查询
    st.subheader("🔍 方式二：输入学号或姓名查询")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_input = st.text_input("请输入学号或姓名", placeholder="例如：G30101 或 陈景明", key="search_input")
    with col2:
        search_button = st.button("🔍 查询", use_container_width=True, key="search_btn")
    
    if search_button and search_input.strip():
        query = search_input.strip()
        result = df[
            (df["学号"].str.contains(query, case=False)) | 
            (df["姓名"].str.contains(query, case=False))
        ]
        
        if len(result) == 1:
            student_row = result.iloc[0]
            st.session_state.selected_student_info = student_row
            create_new_chat(student_row["学号"])
            st.rerun()
        elif len(result) > 1:
            st.markdown("---")
            st.subheader(f"找到 {len(result)} 位匹配的学生，请选择：")
            for _, row in result.iterrows():
                if st.button(f"👤 {row['姓名']} - {row['学号']} - {row['班级']}", key=f"select_{row['学号']}"):
                    st.session_state.selected_student_info = row
                    create_new_chat(row["学号"])
                    st.rerun()
        else:
            st.error(f"未找到学号或姓名包含「{query}」的学生。")

# ==================== 老师查询入口页面 ====================
def show_teacher_search_page():
    """展示老师查询入口页面"""
    st.title("👨‍🏫 老师班级学情分析")
    st.markdown(
        """
        <div style="margin-bottom: 20px; color: #555;">
            请选择您负责的科目，查看班级整体学情分析。
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 返回按钮
    if st.button("🔙 返回首页", key="back_to_entry_teacher"):
        st.session_state.current_role = None
        st.session_state.teacher_subject = None
        st.session_state.teacher_exam = None
        st.session_state.is_head_teacher = False
        st.rerun()
    
    st.markdown("---")
    
    # 是否为班主任
    st.subheader("👔 选择身份类型")
    is_head = st.checkbox("我是班主任（可查看全班各科成绩）")
    st.session_state.is_head_teacher = is_head
    
    st.markdown("---")
    
    if is_head:
        # 班主任：选择班级即可
        st.subheader("📚 选择班级")
        classes = sorted(df["班级"].unique())
        selected_class = st.selectbox("选择班级", classes, key="teacher_class_select")
        
        if st.button("✅ 进入分析", key="confirm_teacher"):
            st.session_state.teacher_subject = "班主任"
            st.session_state.teacher_class = selected_class
            st.rerun()
    else:
        # 普通老师：选择科目
        st.subheader("📚 选择您负责的科目")
        subjects = ["语文", "数学", "英语", "物理", "化学", "生物"]
        selected_subject = st.selectbox("选择科目", subjects, key="teacher_subject_select")
        
        if st.button("✅ 进入分析", key="confirm_teacher"):
            st.session_state.teacher_subject = selected_subject
            st.rerun()

# ==================== 学生详情页面 ====================
def show_student_detail_page():
    """展示学生详情页面"""
    student_row = st.session_state.selected_student_info
    student_id = student_row["学号"]
    
    # 顶部导航
    st.title(f"📊 {student_row['姓名']} 的模考学情分析报告")
    st.markdown(
        f"""
        <div style="margin-bottom: 15px; color: #555;">
            <span style="margin-right: 20px;">📝 学号：{student_row['学号']}</span>
            <span>🏫 班级：{student_row['班级']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 返回按钮
    if st.button("🔙 返回查询页面", key="back_to_search"):
        st.session_state.selected_student_info = None
        st.session_state.current_chat_id = None
        st.session_state.current_messages = []
        st.rerun()
    
    # 考试场次选择
    exam_options = ["一模", "二模", "三模"]
    selected_exam = st.selectbox("📝 选择考试场次", exam_options, key="exam_select_detail")
    
    st.markdown("---")
    
    # 主体布局
    left_col, right_col = st.columns([3, 2])
    
    # ==================== 左侧：成绩与图表 ====================
    with left_col:
        # 1. 成绩表格 - 排名在成绩右侧
        st.subheader(f"📋 {selected_exam} 成绩详情")
        
        # 成绩表格 - 只显示科目和成绩，去掉各科排名
        exams_order = ["一模", "二模", "三模"]
        subjects = ["语文", "数学", "英语", "物理", "化学", "生物"]
        current_idx = exams_order.index(selected_exam)
        prev_exam = exams_order[current_idx - 1] if current_idx > 0 else None
        
        # 使用Streamlit原生表格显示成绩
        score_data = []
        for subj in subjects:
            current_score = student_row[f"{selected_exam}{subj}"]
            prev_score = student_row[f"{prev_exam}{subj}"] if prev_exam else None
            
            # 计算变化
            if prev_exam and not pd.isna(prev_score):
                change = current_score - prev_score
                change_str = f"↑{change}" if change > 0 else f"↓{abs(change)}" if change < 0 else "持平"
            else:
                change_str = "-"
            
            score_data.append({
                "科目": subj,
                "成绩": int(current_score),
                "变化": change_str
            })
        
        # 总分
        current_total = student_row[f"{selected_exam}总分"]
        prev_total = student_row[f"{prev_exam}总分"] if prev_exam else None
        if prev_exam and not pd.isna(prev_total):
            total_change = current_total - prev_total
            total_change_str = f"↑{total_change}" if total_change > 0 else f"↓{abs(total_change)}" if total_change < 0 else "持平"
        else:
            total_change_str = "-"
        
        score_data.append({
            "科目": "总分",
            "成绩": int(current_total),
            "变化": total_change_str
        })
        
        score_df = pd.DataFrame(score_data)
        st.dataframe(score_df, use_container_width=True, hide_index=True)
        
        # 排名信息单独展示
        st.markdown("---")
        st.markdown("**📊 排名信息**")
        class_size = len(df[df['班级'] == student_row['班级']])
        grade_size = len(df)
        
        rank_col1, rank_col2 = st.columns(2)
        with rank_col1:
            # 计算排名变化
            current_class_rank = student_row[f"{selected_exam}班排"]
            prev_class_rank = student_row[f"{prev_exam}班排"] if prev_exam else None
            if prev_exam and not pd.isna(prev_class_rank):
                rank_change = prev_class_rank - current_class_rank
                rank_change_str = f"（进步{rank_change}名）" if rank_change > 0 else f"（退步{abs(rank_change)}名）" if rank_change < 0 else "（持平）"
            else:
                rank_change_str = ""
            st.metric("班级排名", f"第{int(current_class_rank)}名 / 共{class_size}人", rank_change_str)
        
        with rank_col2:
            current_grade_rank = student_row[f"{selected_exam}级排"]
            prev_grade_rank = student_row[f"{prev_exam}级排"] if prev_exam else None
            if prev_exam and not pd.isna(prev_grade_rank):
                rank_change = prev_grade_rank - current_grade_rank
                rank_change_str = f"（进步{rank_change}名）" if rank_change > 0 else f"（退步{abs(rank_change)}名）" if rank_change < 0 else "（持平）"
            else:
                rank_change_str = ""
            st.metric("年级排名", f"第{int(current_grade_rank)}名 / 共{grade_size}人", rank_change_str)
        
        st.markdown("---")
        
        # 2. 当前场次六科成绩对比
        st.subheader(f"📊 {selected_exam} 六科成绩")
        
        subjects = ["语文", "数学", "英语", "物理", "化学", "生物"]
        scores = [student_row[f"{selected_exam}{s}"] for s in subjects]
        
        # 使用柱状图展示六科成绩
        radar_df = pd.DataFrame({
            "科目": subjects,
            "分数": scores
        }).set_index("科目")
        st.bar_chart(radar_df, use_container_width=True)
        
        # 显示各科分数详情
        score_col1, score_col2, score_col3 = st.columns(3)
        for i, (subj, score) in enumerate(zip(subjects, scores)):
            col = [score_col1, score_col2, score_col3][i % 3]
            with col:
                st.metric(subj, f"{score}分")
        
        st.markdown("---")
        
        # 3. 趋势图表 - 修改为按一模、二模、三模顺序展示
        st.subheader("📈 三次模考趋势分析")
        
        exams_list = ["一模", "二模", "三模"]
        
        # 总分趋势（带统计卡片）
        st.markdown("**📊 总分变化趋势**")
        totals = [student_row[f"{e}总分"] for e in exams_list]
        avg_total = round(sum(totals) / len(totals), 1)
        max_total = max(totals)
        min_total = min(totals)
        total_improve = totals[-1] - totals[0]
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("平均总分", f"{avg_total}分")
        with stat_col2:
            st.metric("最高总分", f"{max_total}分")
        with stat_col3:
            st.metric("最低总分", f"{min_total}分")
        with stat_col4:
            st.metric("总分变化", f"{total_improve:+}分")
        
        # 使用数字索引确保顺序正确，然后用文字标注
        total_df = pd.DataFrame({
            "考试阶段": [1, 2, 3],
            "总分": totals,
            "考试名称": exams_list
        })
        st.line_chart(total_df, x="考试阶段", y="总分", use_container_width=True)
        # 添加考试阶段标注说明
        st.markdown(f"<div style='text-align: center; color: #666; font-size: 13px;'>横坐标：1=一模  2=二模  3=三模</div>", unsafe_allow_html=True)
        
        # 排名变化趋势（按一模、二模、三模顺序）
        st.markdown("**📊 排名变化趋势**")
        rank_data = []
        class_size = len(df[df['班级'] == student_row['班级']])
        grade_size = len(df)
        
        for i, exam in enumerate(exams_list):
            class_rank = student_row[f"{exam}班排"]
            grade_rank = student_row[f"{exam}级排"]
            
            # 计算排名变化
            if i == 0:
                class_change = "-"
                grade_change = "-"
            else:
                prev_class_rank = student_row[f"{exams_list[i-1]}班排"]
                prev_grade_rank = student_row[f"{exams_list[i-1]}级排"]
                class_diff = prev_class_rank - class_rank
                grade_diff = prev_grade_rank - grade_rank
                class_change = f"↑{class_diff}" if class_diff > 0 else f"↓{abs(class_diff)}" if class_diff < 0 else "持平"
                grade_change = f"↑{grade_diff}" if grade_diff > 0 else f"↓{abs(grade_diff)}" if grade_diff < 0 else "持平"
            
            rank_data.append({
                "考试": exam,
                "班级排名": f"{int(class_rank)}/{class_size}",
                "班级变化": class_change,
                "年级排名": f"{int(grade_rank)}/{grade_size}",
                "年级变化": grade_change
            })
        
        rank_df = pd.DataFrame(rank_data)
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
        
        # 各科分数变化折线图 - 一门一门展示
        st.markdown("**📊 各科分数变化趋势**")
        for subj in subjects:
            st.markdown(f"**{subj}**")
            subj_scores = [student_row[f"{e}{subj}"] for e in exams_list]
            subj_df = pd.DataFrame({
                "考试阶段": [1, 2, 3],
                "分数": subj_scores
            })
            st.line_chart(subj_df, x="考试阶段", y="分数", use_container_width=True, height=150)
        
        st.markdown("---")
        
        # 4. 单次考试学情分析（底部）
        st.subheader("📝 单次考试学情简析")
        
        def generate_exam_analysis(row, exam):
            """生成单次考试的简要分析"""
            subjects = ["语文", "数学", "英语", "物理", "化学", "生物"]
            scores = {s: row[f"{exam}{s}"] for s in subjects}
            total = row[f"{exam}总分"]
            grade_rank = row[f"{exam}级排"]
            class_rank = row[f"{exam}班排"]
            
            # 找出最高和最低科目
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_subject = sorted_scores[0][0]
            best_score = sorted_scores[0][1]
            worst_subject = sorted_scores[-1][0]
            worst_score = sorted_scores[-1][1]
            
            # 计算年级排名百分比
            rank_percent = round(grade_rank / len(df) * 100, 1)
            
            analysis = f"""
            **{exam}考试总结**：
            
            📌 **整体表现**：总分 **{total}分**，年级排名 **第{grade_rank}名**（年级前 **{rank_percent}%**），班级排名 **第{class_rank}名**。
            
            🏆 **最佳科目**：{best_subject}（{best_score}分），表现优秀，继续保持！
            
            ⚠️ **待提升科目**：{worst_subject}（{worst_score}分），建议加强练习，从基础知识点入手查漏补缺。
            
            💡 **建议**：针对薄弱科目制定专项提升计划，每天安排固定时间复习，建立错题本定期回顾。
            """
            return analysis
        
        st.markdown(generate_exam_analysis(student_row, selected_exam), unsafe_allow_html=True)
    
    # ==================== 右侧：AI智能助手 ====================
    with right_col:
        st.subheader("🤖 AI 智能学习助手")
        
        # 显示当前API配置状态
        api_config = get_api_config()
        if api_config["api_key"]:
            st.success("🔮 AI助手已就绪")
        else:
            st.info("💡 请在左侧侧边栏配置API密钥")
        
        st.markdown("---")
        
        # 对话历史管理
        st.markdown("**对话管理**")
        history_col1, history_col2 = st.columns(2)
        with history_col1:
            if st.button("🆕 新对话", use_container_width=True):
                create_new_chat(student_id)
                st.rerun()
        with history_col2:
            if st.button("🗑️ 清空当前对话", use_container_width=True):
                if st.session_state.current_chat_id:
                    history = load_chat_history(student_id)
                    if st.session_state.current_chat_id in history:
                        history[st.session_state.current_chat_id]["messages"] = []
                        save_chat_history(student_id, history)
                    st.session_state.current_messages = []
                    st.rerun()
        
        # 显示历史对话列表
        history = load_chat_history(student_id)
        if history:
            st.markdown("**历史对话**")
            chat_options = sorted(history.keys(), reverse=True)
            selected_chat = st.selectbox(
                "选择历史对话",
                chat_options,
                format_func=lambda x: f"{history[x]['create_time']} ({len(history[x]['messages'])}条消息)",
                key="history_chat_select"
            )
            if st.button("📂 加载此对话", key="load_history_chat"):
                load_existing_chat(student_id, selected_chat)
                st.rerun()
        
        st.markdown("---")
        
        # 当前对话显示
        st.markdown(f"**当前对话**（创建时间：{history.get(st.session_state.current_chat_id, {}).get('create_time', '新对话')}）")
        
        # 显示历史消息
        for msg in st.session_state.current_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)
        
        st.markdown("---")
        
        # 聊天输入框
        if prompt := st.chat_input("请输入您的问题..."):
            # 添加用户消息
            add_message_to_chat(student_id, "user", prompt)
            
            # 显示用户消息
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            # 生成AI回复
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                full_response = ""
                
                # 获取API配置
                api_config = get_api_config()
                api_key_to_use = api_config["api_key"]
                
                # 如果有API密钥，调用DeepSeek API
                if api_key_to_use:
                    # 流式输出
                    for chunk in call_deepseek_api_stream(
                        prompt, student_row, 
                        api_key_to_use, 
                        api_config["base_url"], 
                        api_config["model"]
                    ):
                        if chunk is None:
                            break
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    # 使用规则引擎
                    full_response = generate_ai_response(prompt, student_row)
                    message_placeholder.markdown(full_response)
                
                # 保存回复
                add_message_to_chat(student_id, "assistant", full_response)
        
        # AI问答逻辑（规则引擎）
        def generate_ai_response(question, row):
            """生成AI回复（规则引擎）"""
            if not question or not question.strip():
                return f"您好！我是{row['姓名']}的专属AI学习助手。我可以帮您分析模考成绩、识别优势与薄弱科目、判断学习趋势并提供针对性建议。请问有什么可以帮助您的？"
            
            q = question.strip()
            subjects = ["语文", "数学", "英语", "物理", "化学", "生物"]
            exams_list = ["一模", "二模", "三模"]
            
            # 识别意图
            mentioned_exams = [e for e in exams_list if e in q]
            matched_subjects = [s for s in subjects if s in q]
            
            ask_strength = any(k in q for k in ["优势", "强项", "最好", "擅长", "突出", "厉害"])
            ask_weak = any(k in q for k in ["薄弱", "弱势", "短板", "最差", "不好", "弱科", "差", "需要提升"])
            ask_trend = any(k in q for k in ["趋势", "走势", "变化", "进步", "退步", "下滑", "提升", "怎么样"])
            ask_report = any(k in q for k in ["报告", "分析", "总结", "概况"])
            ask_advice = any(k in q for k in ["建议", "怎么办", "怎么做", "规划", "策略", "推荐", "帮助", "复习"])
            ask_score = any(k in q for k in ["多少分", "成绩", "分数", "考了多少", "几分"])
            ask_rank = "排名" in q or "第几" in q
            
            # 综合报告
            if ask_report:
                strengths = sorted([(s, row[f"三模{s}"]) for s in subjects], key=lambda x: x[1], reverse=True)[:2]
                weaknesses = sorted([(s, row[f"三模{s}"]) for s in subjects], key=lambda x: x[1])[:2]
                totals = [row[f"{e}总分"] for e in exams_list]
                ranks = [row[f"{e}级排"] for e in exams_list]
                
                report = f"""
**学生概况**
- 姓名：{row['姓名']}
- 班级：{row['班级']}
- 学号：{row['学号']}

**三次模考概览**
{chr(10).join([f'- {e}：总分{totals[i]}分，年级排名{ranks[i]}' for i, e in enumerate(exams_list)])}

**优势科目分析**
{chr(10).join([f'- 🏆 {s[0]}：三模成绩{s[1]}分' for s in strengths])}
建议：保持现有学习方法，可尝试挑战更高难度题目。

**薄弱科目分析**
{chr(10).join([f'- ⚠️ {s[0]}：三模成绩{s[1]}分' for s in weaknesses])}
建议：增加练习时间，从基础知识点开始查漏补缺，建立错题本。

**整体趋势**
- 总分变化：{totals[-1] - totals[0]:+}分
- 排名变化：{ranks[0] - ranks[-1]:+}名

**学习建议**
定期回顾错题，保持规律作息，合理安排各科学习时间，保持积极心态。
                """
                return report
            
            # 成绩查询
            if (matched_subjects or ask_score or ask_rank) and not ask_strength and not ask_weak and not ask_trend and not ask_advice:
                exams_to_show = mentioned_exams if mentioned_exams else ["三模"]
                result = []
                for exam in exams_to_show:
                    part = f"**{exam}成绩详情**\n"
                    if matched_subjects:
                        for subj in matched_subjects:
                            part += f"- {subj}：{row[f'{exam}{subj}']}分\n"
                    else:
                        part += f"- 总分：{row[f'{exam}总分']}分\n"
                    part += f"- 班级排名：{row[f'{exam}班排']}/{len(df[df['班级'] == row['班级']])}\n"
                    part += f"- 年级排名：{row[f'{exam}级排']}/{len(df)}\n"
                    result.append(part)
                return "\n".join(result)
            
            # 优势科目
            if ask_strength:
                strengths = sorted([(s, row[f"三模{s}"]) for s in subjects], key=lambda x: x[1], reverse=True)[:2]
                response = "**优势科目分析**\n\n"
                for s in strengths:
                    response += f"🏆 **{s[0]}**：三模成绩{s[1]}分\n"
                response += "\n💡 建议：继续保持这些科目的学习优势，可尝试拓展难度更高的题目，争取在高考中成为拉分项。"
                return response
            
            # 薄弱科目
            if ask_weak:
                weaknesses = sorted([(s, row[f"三模{s}"]) for s in subjects], key=lambda x: x[1])[:2]
                response = "**薄弱科目分析**\n\n"
                for s in weaknesses:
                    response += f"⚠️ **{s[0]}**：三模成绩{s[1]}分\n"
                response += "\n💡 建议：\n"
                response += "1. 梳理基础知识点，确保概念理解透彻\n"
                response += "2. 建立错题本，定期回顾错误原因\n"
                response += "3. 增加针对性练习，从简单到复杂逐步提升\n"
                response += "4. 寻求老师或同学帮助，及时解决疑难问题"
                return response
            
            # 趋势分析
            if ask_trend:
                totals = [row[f"{e}总分"] for e in exams_list]
                ranks = [row[f"{e}级排"] for e in exams_list]
                total_diff = totals[-1] - totals[0]
                rank_diff = ranks[0] - ranks[-1]
                
                if total_diff > 10 and rank_diff > 5:
                    trend = "进步明显"
                elif total_diff < -10 and rank_diff < -5:
                    trend = "有所下滑"
                else:
                    trend = "相对平稳"
                
                response = f"**整体趋势分析**\n\n📊 当前状态：**{trend}**\n\n"
                response += f"📈 详细变化：\n"
                for i, e in enumerate(exams_list):
                    response += f"- {e}：总分{totals[i]}分，年级排名{ranks[i]}\n"
                response += f"\n从一模到三模，总分变化{total_diff:+}分，排名变化{rank_diff:+}名。\n"
                
                if trend == "进步明显":
                    response += "\n🎉 恭喜！保持这种良好的学习状态，继续努力！"
                elif trend == "有所下滑":
                    response += "\n💪 不要灰心，建议分析近期学习方法，及时调整策略，相信你可以迎头赶上！"
                else:
                    response += "\n⚖️ 成绩稳定是很好的基础，建议设定更高目标，寻求突破。"
                
                return response
            
            # 学习建议
            if ask_advice:
                strengths = sorted([(s, row[f"三模{s}"]) for s in subjects], key=lambda x: x[1], reverse=True)[:2]
                weaknesses = sorted([(s, row[f"三模{s}"]) for s in subjects], key=lambda x: x[1])[:2]
                
                response = "**个性化学习建议**\n\n"
                response += f"📌 **优势科目保持**\n你的优势科目是：{'、'.join([s[0] for s in strengths])}\n建议：保持现有学习方法，适当挑战难题，争取更高分数。\n\n"
                response += f"📌 **薄弱科目提升**\n需要重点关注的科目：{'、'.join([s[0] for s in weaknesses])}\n建议：\n"
                response += "- 每天安排30-60分钟专门攻克薄弱科目\n"
                response += "- 从基础题开始，逐步提升难度\n"
                response += "- 建立错题本，每周回顾\n"
                response += "- 定期进行小测验，检验学习效果\n\n"
                response += "📌 **时间管理建议**\n"
                response += "- 制定详细的学习计划，合理分配各科时间\n"
                response += "- 保持规律作息，保证充足睡眠\n"
                response += "- 适当进行体育锻炼，保持良好状态"
                
                return response
            
            # 打招呼
            if any(k in q for k in ["你好", "Hi", "hello", "您好", "在吗", "嗨"]):
                return f"您好！我是{row['姓名']}的专属AI学习助手。我已经掌握了你三次模考的完整数据，可以为您提供成绩分析、科目诊断和学习建议。请问有什么可以帮助您的吗？"
            
            # 兜底
            return (
                "抱歉，我暂时无法完全理解您的问题。您可以尝试询问：\n\n"
                "📊 **成绩查询**：三模数学多少分？\n"
                "🏆 **科目分析**：我的优势科目是什么？\n"
                "📈 **趋势分析**：整体成绩趋势如何？\n"
                "💡 **学习建议**：给我一些学习建议\n"
                "📋 **综合报告**：生成一份我的学习报告"
            )

# ==================== 老师详情页面 ====================
def show_teacher_detail_page():
    """展示老师班级分析页面"""
    subject = st.session_state.teacher_subject
    is_head = st.session_state.is_head_teacher
    
    if is_head:
        st.title(f"👨‍🏫 班主任班级学情分析 - {st.session_state.teacher_class}")
    else:
        st.title(f"👨‍🏫 {subject}科目班级学情分析")
    
    # 返回按钮
    if st.button("🔙 返回选择科目", key="back_to_teacher_search"):
        st.session_state.teacher_subject = None
        st.session_state.teacher_exam = None
        st.session_state.is_head_teacher = False
        if hasattr(st.session_state, 'teacher_class'):
            del st.session_state.teacher_class
        st.rerun()
    
    st.markdown("---")
    
    # 选择考试场次
    exam_options = ["一模", "二模", "三模"]
    selected_exam = st.selectbox("📝 选择考试场次", exam_options, key="teacher_exam_select")
    st.session_state.teacher_exam = selected_exam
    
    # 获取班级数据
    if is_head:
        class_data = df[df["班级"] == st.session_state.teacher_class]
    else:
        classes = sorted(df["班级"].unique())
        selected_class = st.selectbox("选择班级", classes, key="teacher_class_select")
        class_data = df[df["班级"] == selected_class]
    
    st.markdown("---")
    
    if is_head:
        # 班主任视角：全班各科折线图
        st.subheader(f"📊 {selected_exam} 全班各科成绩分析")
        
        subjects = ["语文", "数学", "英语", "物理", "化学", "生物"]
        
        # 每科一个柱状图展示全班学生成绩
        for subj in subjects:
            st.markdown(f"**📚 {subj}科成绩分布**")
            student_scores = []
            for _, row in class_data.iterrows():
                student_scores.append({
                    "学生": row["姓名"],
                    "分数": row[f"{selected_exam}{subj}"]
                })
            subj_df = pd.DataFrame(student_scores).set_index("学生")
            st.bar_chart(subj_df, use_container_width=True)
            
            # 显示该科目的统计信息
            avg_score = class_data[f"{selected_exam}{subj}"].mean()
            max_score = class_data[f"{selected_exam}{subj}"].max()
            min_score = class_data[f"{selected_exam}{subj}"].min()
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("平均分", f"{round(avg_score, 1)}分")
            with stat_col2:
                st.metric("最高分", f"{max_score}分")
            with stat_col3:
                st.metric("最低分", f"{min_score}分")
            
            st.markdown("---")
        
        # 班级总分分析
        st.subheader(f"📊 {selected_exam} 班级总分分析")
        total_scores = []
        for _, row in class_data.iterrows():
            total_scores.append({
                "学生": row["姓名"],
                "总分": row[f"{selected_exam}总分"]
            })
        total_df = pd.DataFrame(total_scores).set_index("学生")
        st.bar_chart(total_df, use_container_width=True)
        
        avg_total = class_data[f"{selected_exam}总分"].mean()
        max_total = class_data[f"{selected_exam}总分"].max()
        min_total = class_data[f"{selected_exam}总分"].min()
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("平均总分", f"{round(avg_total, 1)}分")
        with stat_col2:
            st.metric("最高总分", f"{max_total}分")
        with stat_col3:
            st.metric("最低总分", f"{min_total}分")
    
    else:
        # 普通老师视角：当前科目分析
        st.subheader(f"📊 {subject}科目 {selected_exam} 成绩分析")
        
        # 当前科目全班成绩柱状图
        st.markdown("**📈 全班学生成绩分布**")
        student_scores = []
        for _, row in class_data.iterrows():
            student_scores.append({
                "学生": row["姓名"],
                "分数": row[f"{selected_exam}{subject}"]
            })
        scores_df = pd.DataFrame(student_scores).set_index("学生")
        st.bar_chart(scores_df, use_container_width=True)
        
        # 统计信息
        avg_score = class_data[f"{selected_exam}{subject}"].mean()
        max_score = class_data[f"{selected_exam}{subject}"].max()
        min_score = class_data[f"{selected_exam}{subject}"].min()
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("平均分", f"{round(avg_score, 1)}分")
        with stat_col2:
            st.metric("最高分", f"{max_score}分")
        with stat_col3:
            st.metric("最低分", f"{min_score}分")
        
        st.markdown("---")
        
        # 选择特定学生查看详细趋势
        st.subheader(f"� 选择学生查看详细成绩趋势")
        student_names = sorted(class_data["姓名"].unique())
        selected_student_name = st.selectbox("选择学生姓名", student_names, key="teacher_student_select")
        
        # 获取选中学生的数据
        selected_student_data = class_data[class_data["姓名"] == selected_student_name].iloc[0]
        
        # 显示该学生的详细趋势分析
        st.markdown(f"### 📊 {selected_student_name} 的成绩趋势分析")
        
        # 当前科目三次模考趋势
        st.markdown(f"**📈 当前科目（{subject}）成绩变化**")
        exams_list = ["一模", "二模", "三模"]
        subject_scores = [selected_student_data[f"{e}{subject}"] for e in exams_list]
        subject_df = pd.DataFrame({
            "考试阶段": [1, 2, 3],
            "分数": subject_scores
        })
        st.line_chart(subject_df, x="考试阶段", y="分数", use_container_width=True)
        st.markdown(f"<div style='text-align: center; color: #666; font-size: 13px;'>横坐标：1=一模  2=二模  3=三模</div>", unsafe_allow_html=True)
        
        # 总分趋势
        st.markdown("**📈 总分变化趋势**")
        totals = [selected_student_data[f"{e}总分"] for e in exams_list]
        total_df = pd.DataFrame({
            "考试阶段": [1, 2, 3],
            "总分": totals
        })
        st.line_chart(total_df, x="考试阶段", y="总分", use_container_width=True)
        st.markdown(f"<div style='text-align: center; color: #666; font-size: 13px;'>横坐标：1=一模  2=二模  3=三模</div>", unsafe_allow_html=True)
        
        # 排名趋势
        st.markdown("**� 排名变化趋势**")
        rank_data = []
        class_size = len(class_data)
        grade_size = len(df)
        
        for i, exam in enumerate(exams_list):
            class_rank = selected_student_data[f"{exam}班排"]
            grade_rank = selected_student_data[f"{exam}级排"]
            
            # 计算排名变化
            if i == 0:
                class_change = "-"
                grade_change = "-"
            else:
                prev_class_rank = selected_student_data[f"{exams_list[i-1]}班排"]
                prev_grade_rank = selected_student_data[f"{exams_list[i-1]}级排"]
                class_diff = prev_class_rank - class_rank
                grade_diff = prev_grade_rank - grade_rank
                class_change = f"↑{class_diff}" if class_diff > 0 else f"↓{abs(class_diff)}" if class_diff < 0 else "持平"
                grade_change = f"↑{grade_diff}" if grade_diff > 0 else f"↓{abs(grade_diff)}" if grade_diff < 0 else "持平"
            
            rank_data.append({
                "考试": exam,
                "班级排名": f"{int(class_rank)}/{class_size}",
                "班级变化": class_change,
                "年级排名": f"{int(grade_rank)}/{grade_size}",
                "年级变化": grade_change
            })
        
        rank_df = pd.DataFrame(rank_data)
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
    
    # 学情分析
    st.markdown("---")
    st.subheader("📝 班级学情简析")
    
    if is_head:
        # 班主任分析
        analysis = f"""
        **{st.session_state.teacher_class} - {selected_exam}分析**：
        
        📌 **班级概况**：
        - 班级人数：{len(class_data)}人
        
        📌 **各科平均分**：
        """
        subjects = ["语文", "数学", "英语", "物理", "化学", "生物"]
        for subj in subjects:
            avg = class_data[f"{selected_exam}{subj}"].mean()
            analysis += f"- {subj}：{round(avg, 1)}分\n"
        
        total_avg = class_data[f"{selected_exam}总分"].mean()
        analysis += f"""
        
        📌 **总分情况**：
        - 班级平均总分：{round(total_avg, 1)}分
        
        💡 **建议**：根据各科成绩分布，针对性地进行辅导，关注成绩较低的学生，帮助他们提升成绩。
        """
    else:
        # 普通老师分析
        subject_avg = class_data[f"{selected_exam}{subject}"].mean()
        max_student = class_data.loc[class_data[f"{selected_exam}{subject}"].idxmax()]["姓名"]
        min_student = class_data.loc[class_data[f"{selected_exam}{subject}"].idxmin()]["姓名"]
        
        # 计算进步/退步情况（与上一次考试对比）
        exams_order = ["一模", "二模", "三模"]
        current_idx = exams_order.index(selected_exam)
        if current_idx > 0:
            prev_exam = exams_order[current_idx - 1]
            improve_count = len(class_data[class_data[f"{selected_exam}{subject}"] > class_data[f"{prev_exam}{subject}"]])
            decline_count = len(class_data[class_data[f"{selected_exam}{subject}"] < class_data[f"{prev_exam}{subject}"]])
            same_count = len(class_data[class_data[f"{selected_exam}{subject}"] == class_data[f"{prev_exam}{subject}"]])
        else:
            improve_count = decline_count = same_count = "N/A"
        
        analysis = f"""
        **{selected_class} - {subject}科目 {selected_exam}分析**：
        
        📌 **整体表现**：
        - 平均分：{round(subject_avg, 1)}分
        - 最高分：{max_score}分（{max_student}）
        - 最低分：{min_score}分（{min_student}）
        
        📌 **与上一次考试对比**：
        - 进步人数：{improve_count}人
        - 退步人数：{decline_count}人
        - 持平人数：{same_count}人
        
        💡 **建议**：重点关注成绩较低的学生，了解他们的学习困难并提供针对性辅导。对于进步较大的学生，可以分享他们的学习方法。
        """
    
    st.markdown(analysis, unsafe_allow_html=True)

# ==================== 主程序入口 ====================
if st.session_state.current_role is None:
    show_entry_page()
elif st.session_state.current_role == "student":
    if st.session_state.selected_student_info is None:
        show_student_search_page()
    else:
        show_student_detail_page()
elif st.session_state.current_role == "teacher":
    if st.session_state.teacher_subject is None:
        show_teacher_search_page()
    else:
        show_teacher_detail_page()