import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import openai
import re

st.set_page_config(page_title="CareerConnect", layout="wide")

# ========== LOGO ===========
LOGO_PATH = "hr.png"  # Chemin du logo local

# ========== ENV ==========
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
openai.api_key = GROQ_API_KEY
openai.base_url = "https://api.groq.com/openai/v1"

# ========== SESSION STATE ==========
if "db_connected" not in st.session_state:
    st.session_state.db_connected = False
if "db_info" not in st.session_state:
    st.session_state.db_info = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = ""
if "last_chart_type" not in st.session_state:
    st.session_state.last_chart_type = "bar"
if "last_df" not in st.session_state:
    st.session_state.last_df = None

def login_page():
    st.title("CAREER-CONNECT")
    st.subheader("Veuillez vous authentifier pour accéder aux autres pages.")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if username == "admin" and password == "admin":
            st.session_state["authenticated"] = True
            st.success("Connexion réussie. Accédez aux autres pages depuis le menu.")
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")

# ========== LLM PROMPT ==========
def get_llm_response(user_input, schema_info):
    # Ajoute l'historique des messages pour la mémoire
    system_prompt = (
        "Tu es un assistant SQL RH. Pour chaque question analytique, réponds STRICTEMENT dans ce format :\n"
        "```sql\nSELECT ...\n```\nType de visualisation : bar/pie/line\nAnalyse : [interprétation claire en français du résultat SQL]\n"
        "NE JAMAIS répondre sans bloc SQL, type de graphique et analyse. NE JAMAIS répondre avec 'SELECT ...' ou une requête incomplète.\n"
        f"Schéma de la base : {schema_info}"
    )
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    # Ajoute l'historique utilisateur/assistant
    for msg in st.session_state.messages[-10:]:  # Garde les 10 derniers échanges
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})
    # Ajoute la question courante
    messages.append({"role": "user", "content": user_input})

    client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    chat = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0
    )
    # Pour usage metrics
    st.session_state.last_usage = getattr(chat, 'usage', None)
    return chat.choices[0].message.content

def extract_sql_and_visu(content):
    sql_query, visu_type, analysis = "", "bar", ""
    # Nettoie les balises HTML éventuelles et les entités
    import re
    from html import unescape
    content = re.sub(r'<[^>]+>', '', content)
    content = unescape(content)
    sql_block = re.search(r'```sql\s*([\s\S]+?)```', content, re.IGNORECASE)
    if sql_block:
        sql_query = sql_block.group(1).strip()
    else:
        # Cas où le SQL n'est pas dans un bloc mais juste après SELECT
        sql_line = re.search(r'SELECT[\s\S]+?(?=Type de visualisation|Analyse|$)', content, re.IGNORECASE)
        if sql_line:
            sql_query = sql_line.group(0).strip()
    # Supprime tous les backticks et balises markdown restants
    sql_query = sql_query.replace('```', '').replace('`', '').strip()
    match = re.search(r'Type de visualisation\s*:\s*(bar|pie|line)', content, re.IGNORECASE)
    if match:
        visu_type = match.group(1).lower()
    analysis_match = re.search(r'Analyse\s*:\s*(.*)', content, re.IGNORECASE)
    if analysis_match:
        analysis = analysis_match.group(1).strip()
    # Correction des noms de colonnes : remplace les underscores par des espaces si le nom existe dans le schéma
    # On suppose que le schéma est accessible via st.session_state.schema_info
    schema_cols = []
    if hasattr(st.session_state, 'schema_info'):
        # Récupère tous les noms de colonnes du schéma
        for line in st.session_state.schema_info.split('\n'):
            if ':' in line:
                cols = line.split(':', 1)[1].split(',')
                for col in cols:
                    col_name = col.split('(')[0].strip()
                    schema_cols.append(col_name)
    # Pour chaque colonne du schéma avec espace, remplace dans le SQL
    for col in schema_cols:
        if ' ' in col:
            # Remplace d'abord les occurrences déjà entre crochets pour éviter [[...]]
            sql_query = re.sub(r'\[+' + re.escape(col) + r'\]+', f'[{col}]', sql_query)
            # Remplace les underscores par espaces puis ajoute une seule paire de crochets
            sql_query = re.sub(r'\b' + re.escape(col.replace(' ', '_')) + r'\b', f'[{col}]', sql_query)
            # Remplace le nom brut par une seule paire de crochets
            sql_query = re.sub(r'(?<!\[)\b' + re.escape(col) + r'\b(?!\])', f'[{col}]', sql_query)
    return sql_query, visu_type, analysis

# ========== GET SCHEMA ==========
def get_schema(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS
        """)).fetchall()
        schema = {}
        value_dict = {}
        for table, col, dtype in result:
            schema.setdefault(table, []).append(f"{col} ({dtype})")
            # Si colonne catégorielle (char, varchar, nvarchar), on extrait les valeurs distinctes (max 100)
            if dtype.lower() in ["char", "varchar", "nvarchar", "nchar", "text"]:
                try:
                    val_query = f"SELECT DISTINCT [{col}] FROM [{table}] WHERE [{col}] IS NOT NULL"
                    vals = conn.execute(text(val_query)).fetchmany(100)
                    value_dict[col.lower()] = set(str(v[0]) for v in vals if v[0] is not None)
                except Exception:
                    pass
    schema_str = "\n".join(f"{t}: {', '.join(cols)}" for t, cols in schema.items())
    # On stocke les valeurs catégorielles dans la session pour la validation
    st.session_state["_distinct_values"] = value_dict
    return schema_str


# ========== PAGE POWER BI ===========
def powerbi_page():
    st.title("Tableau de Bord : Power BI")
    st.info("Si le tableau ne s'affiche pas, ouvrez le lien dans un nouvel onglet.")
    powerbi_url = "https://app.powerbi.com/links/7UnmhC614M?ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730&pbi_source=linkShare"
    st.components.v1.iframe(powerbi_url, width=1140, height=541)
    st.markdown(f"[Ouvrir le tableau dans Power BI]({powerbi_url})", unsafe_allow_html=True)

# ========== PAGE CHATBOT ===========
def chatbot_page():
    st.markdown("""
        <div style='background:#f0f2f6;padding:10px 0 10px 0;text-align:center;font-size:28px;font-weight:bold;border-bottom:2px solid #4F8BF9;color:#4F8BF9;'>🤖 Chatbot Data Visualisation RH</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #e0e0e0;margin:0 0 18px 0;'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,3,1], gap="large")

    with col1:
        st.markdown("<div style='background:#e3f2fd;padding:18px;border-radius:12px;'><span style='font-size:22px;font-weight:bold;color:#1976d2;'>À propos</span></div>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background:#fff;padding:12px 16px;border-radius:8px;margin-top:8px;color:#333;'>
            Ce chatbot RH permet d'analyser dynamiquement votre base de données RH, d'obtenir des requêtes SQL, des analyses automatiques et des visualisations interactives.<br>
            Il comprend aussi le langage naturel et peut répondre à des questions générales ou interpréter des résultats.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid #e0e0e0;margin:18px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='background:#fce4ec;padding:10px 16px;border-radius:8px;margin-top:0;'><span style='font-size:18px;font-weight:bold;color:#d81b60;'>📊 Usage Metrics</span></div>", unsafe_allow_html=True)
        usage = getattr(st.session_state, "last_usage", None)
        if usage:
            st.markdown("<div style='background:#fff;padding:8px 16px;border-radius:8px;margin-top:8px;'>", unsafe_allow_html=True)
            if hasattr(usage, 'total_tokens'):
                st.write(f"**Total Tokens Used**: {usage.total_tokens}")
            if hasattr(usage, 'prompt_tokens'):
                st.write(f"**Prompt Tokens**: {usage.prompt_tokens}")
            if hasattr(usage, 'completion_tokens'):
                st.write(f"**Completion Tokens**: {usage.completion_tokens}")
            price = 0.002 * (getattr(usage, 'total_tokens', 0) / 1000)
            st.write(f"**Total Cost (USD)**: ${price:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div style='background:#e8f5e9;padding:18px;border-radius:12px;'><span style='font-size:22px;font-weight:bold;color:#388e3c;'>💬 Historique</span></div>", unsafe_allow_html=True)
        st.markdown("<div style='background:#fff;padding:12px 16px;border-radius:8px;margin-top:8px;'>", unsafe_allow_html=True)
        for idx, msg in enumerate(st.session_state.messages):
            role = "👤" if msg["role"] == "user" else "🤖"
            if msg["role"] == "assistant":
                sql, visu, analysis = extract_sql_and_visu(msg["content"])
                clean_analysis = re.sub(r'<[^>]+>', '', analysis)
                clean_sql = re.sub(r'```sql\s*([\s\S]+?)```', r'\1', msg["content"], flags=re.IGNORECASE)
                clean_sql = re.sub(r'<[^>]+>', '', clean_sql)
                if sql:
                    st.markdown(f"<span style='font-weight:bold;color:#388e3c;'>{role}</span> : <b>Analyse :</b> {clean_analysis}", unsafe_allow_html=True)
                    st.markdown(f"<b>SQL :</b> <pre style='background:#f8bbd0;padding:8px;border-radius:6px;'>{sql}</pre>", unsafe_allow_html=True)
                else:
                    clean_msg = re.sub(r'<[^>]+>', '', msg['content'])
                    st.markdown(f"<span style='font-weight:bold;color:#388e3c;'>{role}</span> : {clean_msg}", unsafe_allow_html=True)
                if sql:
                    if st.button(f"Exécuter ce résultat", key=f"exec_hist_{idx}"):
                        with col2:
                            with st.chat_message("assistant"):
                                st.markdown(f"<b>Analyse :</b> {clean_analysis}", unsafe_allow_html=True)
                                st.markdown(f"<b>SQL :</b> <pre style='background:#f8bbd0;padding:8px;border-radius:6px;'>{sql}</pre>", unsafe_allow_html=True)
                                st.markdown(f"<b>Visualisation :</b> {visu}", unsafe_allow_html=True)
                                try:
                                    DB_SERVER = os.getenv("DB_SERVER")
                                    DB_NAME = os.getenv("DB_NAME")
                                    DB_USER = os.getenv("DB_USER")
                                    DB_PASSWORD = os.getenv("DB_PASSWORD")
                                    conn_str = f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
                                    engine = create_engine(conn_str)
                                    with engine.connect() as conn:
                                        df = pd.read_sql_query(sql, conn.connection)
                                    st.session_state.last_df = df
                                    st.dataframe(df)
                                    fig = None
                                    if visu == "pie" and df.shape[1] >= 2:
                                        fig = px.pie(df, names=df.columns[0], values=df.columns[1])
                                    elif visu == "line" and df.shape[1] >= 2:
                                        fig = px.line(df, x=df.columns[0], y=df.columns[1])
                                    elif visu == "bar" and df.shape[1] >= 2:
                                        fig = px.bar(df, x=df.columns[0], y=df.columns[1])
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("Aucune visualisation possible avec ce résultat.")
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de l'exécution : {e}")
            else:
                clean_msg = re.sub(r'<[^>]+>', '', msg['content'])
                st.markdown(f"<span style='font-weight:bold;color:#388e3c;'>{role}</span> : {clean_msg}", unsafe_allow_html=True)
            st.markdown("<hr style='border:1px dashed #bdbdbd;margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        DB_SERVER = os.getenv("DB_SERVER")
        DB_NAME = os.getenv("DB_NAME")
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        conn_str = f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
        engine = create_engine(conn_str)
        schema_info = get_schema(engine)
        st.session_state.schema_info = schema_info

        prompt = st.chat_input("Posez une question ou discutez 👇")
        if prompt:
            def detect_intent_llm(question):
                """
                Utilise le LLM pour classifier la question :
                - Retourne 'analytique' si la question nécessite une requête SQL sur les données RH
                - Retourne 'conversationnelle' sinon (greetings, explication, etc)
                """
                intent_prompt = (
                    "Tu es un assistant RH. Pour la question suivante, réponds uniquement par 'analytique' si elle nécessite une requête SQL sur les données RH, sinon réponds uniquement par 'conversationnelle'.\n"
                    f"Question : {question}"
                )
                client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
                chat = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": intent_prompt}],
                    temperature=0
                )
                return chat.choices[0].message.content.strip().lower()

            # Détection intelligente de l'intention
            intent = detect_intent_llm(prompt)
            if intent == "conversationnelle":
                # Ajoute d'abord le message utilisateur à l'historique
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Réponse conversationnelle avec mémoire du contexte
                # On envoie l'historique des 10 derniers échanges au LLM
                messages = []
                for msg in st.session_state.messages[-10:]:
                    if msg["role"] == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        messages.append({"role": "assistant", "content": msg["content"]})
                client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
                chat = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=messages,
                    temperature=0
                )
                content = chat.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": content})
                # Ne pas afficher ici, l'affichage se fait via l'historique
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                content = get_llm_response(prompt, schema_info)
                sql, visu, analysis = extract_sql_and_visu(content)
                st.session_state.last_sql = sql
                st.session_state.last_analysis = analysis
                st.session_state.last_chart_type = visu
                st.session_state.last_content = content

                # Vérification des colonnes utilisées dans la requête SQL
                def get_all_schema_columns(schema_info):
                    cols = set()
                    for line in schema_info.split('\n'):
                        if ':' in line:
                            colparts = line.split(':', 1)[1].split(',')
                            for col in colparts:
                                col_name = col.split('(')[0].strip()
                                cols.add(col_name.lower())
                    return cols

                def get_columns_in_sql(sql):
                    import re
                    # Récupère les alias définis dans le SELECT (AS ...)
                    alias_matches = re.findall(r'AS\s+([\w\[\]]+)', sql, flags=re.IGNORECASE)
                    aliases = set(a.strip('[]').lower() for a in alias_matches)
                    # On retire les alias (après AS ...)
                    sql_no_alias = re.sub(r'AS\s+\w+', '', sql, flags=re.IGNORECASE)
                    # On retire les noms de tables (après FROM/JOIN ...)
                    sql_no_tables = re.sub(r'(FROM|JOIN)\s+\w+', '', sql_no_alias, flags=re.IGNORECASE)
                    # On retire les fonctions SQL (ex: GETDATE(), YEAR(...))
                    sql_no_funcs = re.sub(r'\b\w+\s*\([^)]*\)', '', sql_no_tables)
                    tokens = re.findall(r'\b\w+\b', sql_no_funcs)
                    reserved = {"select", "from", "where", "group", "by", "order", "join", "on", "as", "and", "or", "desc", "asc", "count", "sum", "avg", "min", "max", "distinct"}
                    # Récupère les valeurs catégorielles connues
                    value_dict = st.session_state.get("_distinct_values", {})
                    all_values = set()
                    for vset in value_dict.values():
                        all_values.update(vset)
                    # Ignore les nombres, les mots réservés, les valeurs catégorielles connues et les alias SQL (insensibles à la casse)
                    return [t for t in tokens if t.lower() not in reserved and not t.isdigit() and t.lower() not in (val.lower() for val in all_values) and t.lower() not in aliases]

                schema_cols = get_all_schema_columns(schema_info)
                sql_cols = get_columns_in_sql(sql)
                # On ne vérifie que les colonnes, pas les tables
                missing_cols = [col for col in sql_cols if col.lower() not in schema_cols]
                if missing_cols:
                    error_msg = f"❌ Erreur : Les colonnes suivantes n'existent pas dans la base : {', '.join(missing_cols)}.\nVeuillez reformuler votre question ou vérifier le schéma."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": content})

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_content = st.session_state.messages[-1]["content"]
            sql, visu, analysis = extract_sql_and_visu(last_content)
            if not sql:
                with st.chat_message("assistant"):
                    clean_content = re.sub(r'<[^>]+>', '', last_content)
                    st.markdown(clean_content)
            else:
                with st.chat_message("assistant"):
                    clean_analysis = re.sub(r'<[^>]+>', '', analysis)
                    st.markdown(f"**Analyse :** {clean_analysis}")
                    st.markdown(f"```sql\n{sql}\n```")
                    st.markdown(f"**Visualisation :** {visu}")

                    if sql:
                        if st.button("Exécuter la requête", key=f"exec_{len(st.session_state.messages)}"):
                            try:
                                with engine.connect() as conn:
                                    df = pd.read_sql_query(sql, conn.connection)
                                st.session_state.last_df = df
                                st.dataframe(df)
                                fig = None
                                if visu == "pie" and df.shape[1] >= 2:
                                    fig = px.pie(df, names=df.columns[0], values=df.columns[1])
                                elif visu == "line" and df.shape[1] >= 2:
                                    fig = px.line(df, x=df.columns[0], y=df.columns[1])
                                elif visu == "bar" and df.shape[1] >= 2:
                                    fig = px.bar(df, x=df.columns[0], y=df.columns[1])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Aucune visualisation possible avec ce résultat.")
                            except Exception as e:
                                st.error(f"❌ Erreur lors de l'exécution : {e}")

        if st.session_state.last_df is not None:
            st.markdown("---\n### Dernier résultat :")
            st.dataframe(st.session_state.last_df)


# ========== PAGE DASHBOARD RH ===========
def dashboard_rh_page():
    st.title("Dashboard RH Interactif")
    DB_SERVER = os.getenv("DB_SERVER")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    conn_str = f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(conn_str)
    # Affiche la structure des tables pour aider à corriger la requête
    try:
        with engine.connect() as conn:
            tables = pd.read_sql_query("""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                ORDER BY TABLE_NAME
            """, conn.connection)
        st.subheader("Structure des tables disponibles")
        st.dataframe(tables)
        # Suggestion automatique d'un graphique si possible
        # Recherche une table avec au moins deux colonnes numériques ou catégorielles
        table_names = tables['TABLE_NAME'].unique()
        selected_table = st.selectbox("Choisissez une table pour visualiser les données :", table_names)
        table_cols = tables[tables['TABLE_NAME'] == selected_table]['COLUMN_NAME'].tolist()
        st.write(f"Colonnes de la table {selected_table} : {table_cols}")
        # Permet à l'utilisateur de choisir les colonnes pour X et Y
        x_col = st.selectbox("Colonne X (catégorie)", table_cols)
        y_col = st.selectbox("Colonne Y (valeur)", table_cols)
        # Affiche le graphique si possible
        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn.connection)
            st.subheader(f"Graphique : {y_col} par {x_col}")
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} par {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df[[x_col, y_col]])
        except Exception as e:
            st.error(f"Impossible d'afficher le graphique : {e}")
    except Exception as e:
        st.error(f"Erreur lors de la récupération de la structure des tables : {e}")

# ========== NAVIGATION ===========
def main():
    st.sidebar.image(LOGO_PATH, width=150)
    st.sidebar.title("CareerConnect")
    menu = st.sidebar.selectbox(
        "Menu",
        ["Authentification", "Chatbot", "Power BI", "Dashboard RH"]
    )
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if menu == "Authentification":
        login_page()
    elif menu == "Chatbot":
        if st.session_state["authenticated"]:
            chatbot_page()
        else:
            st.warning("Veuillez vous connecter pour accéder à cette page.")
    elif menu == "Power BI":
        if st.session_state["authenticated"]:
            powerbi_page()
        else:
            st.warning("Veuillez vous connecter pour accéder à cette page.")
    elif menu == "Dashboard RH":
        if st.session_state["authenticated"]:
            dashboard_rh_page()
        else:
            st.warning("Veuillez vous connecter pour accéder à cette page.")

# ========== APP LAUNCH ========== 
if __name__ == "__main__":
    main()