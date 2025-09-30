# streamlit run app.py
# Requisiti:
#   pip install streamlit apify-client pandas python-dotenv openai
# ENV: APIFY_TOKEN (per scraping/enrichment)
#      OPENAI_API_KEY (per il conteggio università via AI)

import os, json, re
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import streamlit as st
from apify_client import ApifyClient
from openai import OpenAI
import altair as alt

st.set_page_config(page_title="UV Deal Sourcing Tool", layout="wide")

# ----------------------------
# Utils
# ----------------------------
def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def normalize_linkedin_url(u: str) -> str:
    try:
        from urllib.parse import urlparse
        p = urlparse(u)
        scheme = "https"
        host = (p.hostname or "").lower()
        host = (host
                .replace("www.linkedin.", "linkedin.")
                .replace("it.linkedin.", "linkedin.")
                .replace("es.linkedin.", "linkedin.")
                .replace("uk.linkedin.", "linkedin."))
        path = (p.path or "").rstrip("/")
        return f"{scheme}://{host}{path}"
    except Exception:
        return (u or "").strip()

def download_dataset_items(client: ApifyClient, dataset_id: str) -> List[Dict[str, Any]]:
    items = []
    for it in client.dataset(dataset_id).iterate_items():
        items.append(it)
    return items

def items_to_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items: return pd.DataFrame()
    rows = []
    for it in items:
        for r in it.get("organicResults", []):
            rows.append({
                "page": it.get("searchQuery", {}).get("page"),
                "searchTerm": it.get("searchQuery", {}).get("term"),
                "position": r.get("position"),
                "title": r.get("title"),
                "url": r.get("url") or r.get("link") or r.get("sourceUrl"),
                "displayedUrl": r.get("displayedUrl"),
                "snippet": r.get("description") or r.get("snippet"),
                "location": (r.get("personalInfo") or {}).get("location"),
                "followers": r.get("followersAmount"),
            })
    return pd.DataFrame(rows)

def split_name_from_title(title: str) -> Tuple[str, str]:
    if not title: return "", ""
    first_part = re.split(r"\s[-–—|·•]\s", title, maxsplit=1)[0]
    first_part = re.sub(r"\s+\([^)]*\)$", "", first_part).strip()
    first_part = re.sub(r"^\s*LinkedIn\s*›\s*", "", first_part, flags=re.IGNORECASE).strip()
    tokens = [t for t in re.split(r"\s+", first_part) if t]
    if len(tokens) >= 2: return tokens[0], " ".join(tokens[1:])
    return first_part, ""

def build_people_table(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for page in items:
        for r in page.get("organicResults", []):
            url = r.get("url") or r.get("link") or r.get("sourceUrl")
            if not url or "linkedin.com" not in url: continue
            title = r.get("title") or ""
            snippet = r.get("description") or r.get("snippet") or ""
            location = (r.get("personalInfo") or {}).get("location")
            followers = r.get("followersAmount")
            nome, cognome = split_name_from_title(title)
            url_norm = normalize_linkedin_url(url)
            rows.append({
                "Nome": nome,
                "Cognome": cognome,
                "Title": title,
                "Snippet": snippet,
                "Location": location,
                "Followers": followers,
                "LinkedIn": url_norm,
            })
    df = pd.DataFrame(rows, columns=["Nome","Cognome","Title","Snippet","Location","Followers","LinkedIn"])
    if not df.empty: df = df.drop_duplicates(subset=["LinkedIn"]).reset_index(drop=True)
    return df

# ----------------------------
# Parsing caption (date/durata) per esperienze
# ----------------------------
MONTHS = {
    # EN
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,
    "jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
    "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12,
    # IT
    "gen":1,"gennaio":1,"febbraio":2,"marzo":3,"aprile":4,"maggio":5,"giu":6,"giugno":6,
    "lug":7,"luglio":7,"ago":8,"agosto":8,"set":9,"settembre":9,
    "ott":10,"ottobre":10,"novembre":11,"dic":12,"dicembre":12,
}
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DUR_RE = re.compile(
    r"(?:(?P<yrs>\d+)\s*(?:yrs?|anni|year|years|anno))?\s*(?:[,/·]?\s*)?(?:(?P<mos>\d+)\s*(?:mos?|mesi|month|months|mese))?",
    re.IGNORECASE
)

def _parse_month_year(token: str) -> Tuple[Optional[int], Optional[int]]:
    t = token.strip().lower()
    parts = re.split(r"[ \-_/.,]", t)
    if not parts: return None, None
    m = None; y = None
    for part in parts:
        if part in MONTHS: m = MONTHS[part]; break
    y4 = YEAR_RE.search(t)
    if y4: y = int(y4.group())
    else:
        m2 = re.search(r"\b(\d{2})\b", t)
        if m2:
            yy = int(m2.group(1))
            y = 2000 + yy if yy <= 40 else 1900 + yy
    return m, y

def parse_caption_dates(caption: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
    if not caption: return None, None, None
    main = caption.split("·")[0].strip()
    parts = re.split(r"\s*[-–—]\s*", main)
    if len(parts) == 1:
        m = YEAR_RE.search(parts[0]); 
        return (f"{int(m.group()):04d}-01-01" if m else None, None, None)
    start_str, end_str = parts[0], parts[1]
    if re.search(r"(present|presente|oggi|attuale)", end_str, re.IGNORECASE):
        end_ymd, is_curr = None, True
    else:
        em, ey = _parse_month_year(end_str)
        if ey: end_ymd = f"{ey:04d}-{(em or 1):02d}-01"
        else:
            m = YEAR_RE.search(end_str); end_ymd = f"{int(m.group()):04d}-01-01" if m else None
        is_curr = False if end_ymd else None
    sm, sy = _parse_month_year(start_str)
    if sy: start_ymd = f"{sy:04d}-{(sm or 1):02d}-01"
    else:
        m = YEAR_RE.search(start_str); start_ymd = f"{int(m.group()):04d}-01-01" if m else None
    return start_ymd, end_ymd, is_curr

def parse_duration_months(text: Optional[str]) -> Optional[int]:
    if not text: return None
    m = DUR_RE.search(text)
    if not m: return None
    total = 0
    if m.group("yrs"): total += int(m.group("yrs")) * 12
    if m.group("mos"): total += int(m.group("mos"))
    return total or None

def guess_employment_type(text: Optional[str]) -> Optional[str]:
    if not text: return None
    t = text.lower()
    for lab in ["full-time","part-time","intern","internship","contract","freelance","self-employed","co-founder","founder","apprenticeship","volunteer",
                "tempo pieno","tempo parziale","tirocin","contratto","autonom","volontar"]:
        if lab in t: return lab
    return None

def fmt_ymd(ymd: Optional[str]) -> Optional[str]:
    if not ymd: return None
    m = re.match(r"(\d{4})-(\d{2})-\d{2}", ymd)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return ymd

# ----------------------------
# Enrichment
# ----------------------------
def run_linkedin_enrichment(apify_client: ApifyClient, profile_urls: list[str], batch_size: int = 80) -> list[dict]:
    all_items = []
    if not profile_urls: return all_items
    chunks = [profile_urls[i:i+batch_size] for i in range(0, len(profile_urls), batch_size)]
    progress = st.progress(0.0, text="Invio batch ad Apify…")
    for i, ch in enumerate(chunks, start=1):
        run_input = {"profileUrls": ch}
        run = apify_client.actor("2SyF0bVxmgGr8IVCZ").call(run_input=run_input)
        for it in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
            all_items.append(it)
        progress.progress(i / len(chunks), text=f"Completato batch {i}/{len(chunks)}")
    progress.empty()
    return all_items

# ----------------------------
# Costruzione People Master
# ----------------------------
def summarize_skills(item: dict, top_n: int = 8) -> Optional[str]:
    titles = [s.get("title") for s in (item.get("skills") or []) if s.get("title")]
    if titles: return ", ".join(titles[:top_n])
    return item.get("topSkillsByEndorsements")

def experiences_full_string(item: dict) -> Optional[str]:
    exps = item.get("experiences") or []
    if not exps:
        return None

    def _desc_from(sc_list):
        texts = []
        for sc in sc_list or []:
            for d in sc.get("description") or []:
                t = d.get("text")
                if t:
                    texts.append(t.strip())
        if not texts:
            return None
        first = texts[0].split("\n")[0]
        return first[:300]

    lines = []
    for exp in exps:
        breakdown = exp.get("breakdown")
        if breakdown:
            company = exp.get("title")
            exp_caption = exp.get("caption") or ""
            for sc in exp.get("subComponents") or []:
                role = sc.get("title")
                caption = sc.get("caption") or exp_caption
                loc = sc.get("metadata") or exp.get("metadata") or ""
                start_d, end_d, is_curr = parse_caption_dates(caption)
                start_s = fmt_ymd(start_d) or ""
                end_s = fmt_ymd(end_d) if end_d else ("Present" if is_curr else "")
                when = f"{start_s} → {end_s}".strip()
                desc = _desc_from([sc]) or ""
                part = f"{when}: {role or ''} @ {company or ''}".strip()
                if loc:
                    part += f" — {loc}"
                if desc:
                    part += f" | {desc}"
                lines.append(part)
        else:
            role = exp.get("title")
            company = exp.get("subtitle") or exp.get("title")
            caption = exp.get("caption") or ""
            loc = exp.get("metadata") or ""
            start_d, end_d, is_curr = parse_caption_dates(caption)
            start_s = fmt_ymd(start_d) or ""
            end_s = fmt_ymd(end_d) if end_d else ("Present" if is_curr else "")
            when = f"{start_s} → {end_s}".strip()
            desc = _desc_from(exp.get("subComponents")) or ""
            part = f"{when}: {role or ''} @ {company or ''}".strip()
            if loc:
                part += f" — {loc}"
            if desc:
                part += f" | {desc}"
            lines.append(part)

    return " • ".join([l for l in lines if l])

def extract_experience_blocks(item: dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    exps = item.get("experiences") or []
    if not exps: return None, None, None, None, None, None

    def _desc_list(sc_list):
        texts = []
        for sc in sc_list or []:
            for d in sc.get("description") or []:
                t = d.get("text"); 
                if t: texts.append(t)
        return texts

    current_role = current_company = current_start = current_loc = current_dur = None
    lines = []
    for idx, exp in enumerate(exps):
        breakdown = exp.get("breakdown")
        if breakdown:
            company = exp.get("title")
            for j, sc in enumerate(exp.get("subComponents") or []):
                role = sc.get("title")
                caption = sc.get("caption") or ""
                loc = sc.get("metadata") or exp.get("metadata") or ""
                start_d, end_d, _ = parse_caption_dates(caption or exp.get("caption"))
                when = caption or (exp.get("caption") or "")
                desc = " ".join(_desc_list([sc])) or ""
                line = f"{when}: {role} @ {company}" + (f" — {loc}" if loc else "")
                if desc:
                    line += f" | {desc}"
                lines.append(line)
                if idx == 0 and j == 0:
                    current_role = role
                    current_company = company
                    current_start = start_d
                    current_loc = loc or None
                    current_dur = caption or exp.get("caption") or None
        else:
            role = exp.get("title")
            company = exp.get("subtitle") or exp.get("title")
            caption = exp.get("caption") or ""
            loc = exp.get("metadata") or ""
            desc = " ".join(_desc_list(exp.get("subComponents"))) or ""
            line = f"{caption}: {role} @ {company}" + (f" — {loc}" if loc else "")
            if desc:
                line += f" | {desc}"
            lines.append(line)
            if idx == 0:
                current_role = role
                current_company = company
                current_start, _, _ = parse_caption_dates(caption)
                current_loc = loc or None
                current_dur = caption or None
    timeline_text = " • ".join([l for l in lines if l])
    return current_role, current_company, current_start, current_loc, current_dur, timeline_text

def extract_education_blocks(item: dict) -> Tuple[Optional[str], Optional[str]]:
    edus = item.get("educations") or []
    if not edus: return None, None

    def _short_degree(subtitle: str) -> Tuple[Optional[str], Optional[str]]:
        if not subtitle: return None, None
        parts = [p.strip() for p in subtitle.split(",")]
        degree = parts[0] if parts else None
        field = parts[1] if len(parts) > 1 else None
        return degree, field

    lines = []
    top = None
    for i, ed in enumerate(edus):
        school = ed.get("title")
        degree, field = _short_degree(ed.get("subtitle") or "")
        caption = ed.get("caption") or ""
        notes = []
        for sc in ed.get("subComponents") or []:
            for d in sc.get("description") or []:
                t = d.get("text")
                if t: notes.append(t)
        note = "; ".join(notes) if notes else None
        line = f"{caption}: {degree or ''}{(' in ' + field) if field else ''} @ {school}"
        if note:
            line += f" | {note}"
        lines.append(line.strip())
        if i == 0:
            short = f"{degree or ''}{(' in ' + field) if field else ''}".strip()
            top = f"{school} — {short}" if short else school
    education_text = " • ".join([l for l in lines if l])
    return top, education_text

def person_master_row(item: dict) -> dict:
    current_role, current_company, current_start, current_loc, current_dur, exp_timeline = extract_experience_blocks(item)
    edu_top, edu_text = extract_education_blocks(item)
    return {
        "fullName": item.get("fullName"),
        "headline": item.get("headline"),
        "location": item.get("addressWithCountry") or item.get("addressWithoutCountry") or item.get("addressCountryOnly"),
        "current_role": current_role,
        "current_company": current_company,
        "current_start": current_start,
        "current_location": current_loc,
        "current_duration": current_dur,
        "experiences_full": experiences_full_string(item),
        "education_top": edu_top,
        "education_text": edu_text,
        "skills": summarize_skills(item, top_n=8),
        "connections": item.get("connections"),
        "followers": item.get("followers"),
        "email": item.get("email"),
        "mobileNumber": item.get("mobileNumber"),
        "linkedinUrl": item.get("linkedinUrl"),
        "companyLinkedin": item.get("companyLinkedin"),
        "profilePicHighQuality": item.get("profilePicHighQuality"),
    }

def build_people_master(items: List[dict]) -> pd.DataFrame:
    rows = [person_master_row(it) for it in items]
    df = pd.DataFrame(rows)
    col_order = [
        "fullName","headline","location",
        "current_role","current_company","current_start","current_location","current_duration",
        "experiences_full",
        "education_top","education_text",
        "skills","connections","followers","email","mobileNumber",
        "linkedinUrl","companyLinkedin","profilePicHighQuality",
    ]
    for c in col_order:
        if c not in df.columns: df[c] = None
    return df[col_order]

# ----------------------------
# OpenAI helper (conteggi università)
# ----------------------------
def _safe_json_extract(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"error": "JSON non valido", "raw": s[:1000]}
    return {"error": "Nessun JSON trovato", "raw": s[:500]}

def count_universities_with_openai(df: pd.DataFrame, api_key: str, model: str = "gpt-4o-mini") -> dict:
    if "education_top" not in df.columns:
        return {"error": "Colonna 'education_top' assente nel DataFrame"}

    values = df["education_top"].dropna().astype(str).tolist()
    if not values:
        return {"error": "Nessun valore in 'education_top'"}

    MAX_ROWS = 2000
    rows = values[:MAX_ROWS]

    client = OpenAI(api_key=api_key)

    system_msg = (
        "Sei un data analyst. Riceverai una lista di righe dalla colonna 'education_top'. "
        "Normalizza i nomi delle università (es. 'Harvard', 'Harvard Univ.' -> 'Harvard University') e conta le occorrenze. "
        "Ignora degree/field e voci non-accademiche. "
        "Rispondi SOLO con JSON valido:\n"
        "{\n"
        '  "universities": {"<Nome Università Normalizzato>": <conteggio>, ...},\n'
        '  "total_rows": <numero_righe_processate>,\n'
        '  "unique_universities": <numero_università_distinte>\n'
        "}"
    )
    user_msg = "Righe:\n" + "\n".join(f"- {r}" for r in rows)

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = resp.choices[0].message.content if resp and resp.choices else ""
    data = _safe_json_extract(content)
    if isinstance(data, dict):
        data.setdefault("total_rows", len(rows))
        if "universities" in data and isinstance(data["universities"], dict):
            data.setdefault("unique_universities", len(data["universities"]))
    return data

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.image("logo2.png", width='stretch')
st.sidebar.header("🔐 Credenziali")
APIFY_TOKEN = st.sidebar.text_input("APIFY_TOKEN", get_env("APIFY_TOKEN", ""), type="password")

st.sidebar.header("🧠 OpenAI")
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY", get_env("OPENAI_API_KEY", ""), type="password")
OPENAI_MODEL = st.sidebar.text_input("MODEL", get_env("OPENAI_MODEL", "gpt-4o-mini"))

# ----------------------------
# Tabs
# ----------------------------
tab_scrape, tab_upload_json, tab_upload_csv = st.tabs([
    "▶️ SERP → People",
    "📤 Carica JSON SERP",
    "📥 Carica CSV People"
])

# ===== TAB 1: SERP
with tab_scrape:
    #st.image("logo2.png",  width='stretch')
    keywords = ["stealth", "stealth mode", "founder", "co-founder", "investor", "CEO", "CTO"]

    selected_keywords = st.sidebar.multiselect(
        "📌 Parole chiave da includere",
        options=keywords,
        default=["stealth","stealth mode", "founder", "co-founder"],
        help="Seleziona i termini da cercare nel profilo"
    )

    country_code = st.sidebar.selectbox(
        "🌍 Seleziona Paese",
        options=["it", "fr", "de", "es", "uk", "us"],
        index=0
    )

    site_filter = st.sidebar.radio(
        "🌐 Filtra dominio",
        options=[ "it.linkedin.com", "linkedin.com","www.linkedin.com"],
        index=0
    )

    max_pages = st.sidebar.slider(
        "📄 Numero massimo di pagine (10 risultati/pagina)",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )

    if selected_keywords:
        keywords_query = " OR ".join([f'"{kw}"' for kw in selected_keywords])
    else:
        keywords_query = ""

    query_text = f'site:{site_filter}/in ({keywords_query}) {country_code.upper()}'

    st.sidebar.markdown("---")
    st.sidebar.write("⚙️ Parametri pronti per la ricerca:")
    st.sidebar.code(query_text, language="bash")

    if st.button("🚀 Avvia scraping"):
        if not APIFY_TOKEN:
            st.error("Imposta APIFY_TOKEN nella sidebar.")
        else:
            apify = ApifyClient(APIFY_TOKEN)
            run = apify.actor("apify/google-search-scraper").call(run_input={
                "queries": query_text,
                "countryCode": country_code,
                "maxPagesPerQuery": int(max_pages),
                "site": site_filter,
            })
            dataset_id = run["defaultDatasetId"]
            items = download_dataset_items(apify, dataset_id)
            st.session_state["serp_items"] = items
            st.success(f"Scaricate {len(items)} pagine SERP.")
            st.download_button(
                "⬇️ Scarica JSON SERP",
                json.dumps(items, ensure_ascii=False, indent=2),
                "google_serp_results.json",
                "application/json",
                key="dl_json_serp_scrape"
            )

    if st.session_state.get("serp_items"):
        df_people = build_people_table(st.session_state["serp_items"])
        st.subheader("📋 People (Locale) dalla SERP")
        st.dataframe(df_people, width='stretch', height=360,
                     column_config={"LinkedIn": st.column_config.LinkColumn("LINK", display_text="Apri profilo")})
        st.download_button(
            "⬇️ Scarica CSV (People)",
            df_people.to_csv(index=False).encode("utf-8"),
            "people_table.csv",
            "text/csv",
            key="dl_people_scrape"
        )

        if st.button("🚀 PRENDI INFO SULLE PERSONE (Enrichment)"):
            if not APIFY_TOKEN:
                st.error("Imposta APIFY_TOKEN.")
            else:
                apify = ApifyClient(APIFY_TOKEN)
                urls = df_people["LinkedIn"].dropna().map(normalize_linkedin_url).drop_duplicates().tolist()
                with st.spinner(f"Enrichment di {len(urls)} profili…"):
                    enriched = run_linkedin_enrichment(apify, urls)
                st.session_state["enriched_items"] = enriched
                df_master = build_people_master(enriched)
                st.session_state["people_master"] = df_master
                st.success(f"Creato People Master con {len(df_master)} righe.")
                st.download_button(
                    "⬇️ Scarica CSV (People Master)",
                    df_master.to_csv(index=False).encode("utf-8"),
                    "people_master.csv",
                    "text/csv",
                    key="dl_master_from_scrape"
                )

# ===== TAB 2: Upload JSON SERP
with tab_upload_json:
    st.header("Carica JSON SERP già scaricato")
    file = st.file_uploader("Carica file JSON della SERP", type=["json"])
    if file:
        try:
            items = json.load(file)
            st.session_state["serp_items"] = items
            st.success(f"Caricate {len(items)} pagine SERP dal file.")
        except Exception as e:
            st.error(f"Errore: {e}")

    if st.session_state.get("serp_items"):
        df_people = build_people_table(st.session_state["serp_items"])
        st.subheader("📋 People (Locale) dal file caricato")
        st.dataframe(df_people, width='stretch', height=360,
                     column_config={"LinkedIn": st.column_config.LinkColumn("LINK", display_text="Apri profilo")})
        st.download_button(
            "⬇️ Scarica CSV (People)",
            df_people.to_csv(index=False).encode("utf-8"),
            "people_table.csv",
            "text/csv",
            key="dl_people_upload_json"
        )

        if st.button("🚀 PRENDI INFO SULLE PERSONE (Enrichment)", key="btn_enrich_upload_json"):
            if not APIFY_TOKEN:
                st.error("Imposta APIFY_TOKEN.")
            else:
                apify = ApifyClient(APIFY_TOKEN)
                urls = df_people["LinkedIn"].dropna().map(normalize_linkedin_url).drop_duplicates().tolist()
                with st.spinner(f"Enrichment di {len(urls)} profili…"):
                    enriched = run_linkedin_enrichment(apify, urls)
                st.session_state["enriched_items"] = enriched
                df_master = build_people_master(enriched)
                st.session_state["people_master"] = df_master
                st.success(f"Creato People Master con {len(df_master)} righe.")
                st.download_button(
                    "⬇️ Scarica CSV (People Master)",
                    df_master.to_csv(index=False).encode("utf-8"),
                    "people_master.csv",
                    "text/csv",
                    key="dl_master_from_upload_json"
                )

# ===== TAB 3: Upload CSV People (con colonna LinkedIn/linkedinUrl)
with tab_upload_csv:
    st.header("Carica CSV People (colonna LinkedIn o linkedinUrl)")
    csvf = st.file_uploader("Carica CSV", type=["csv"])
    if csvf:
        try:
            df_csv = pd.read_csv(csvf)
            link_col = None
            for c in df_csv.columns:
                if c.lower() in ("linkedin","linkedinurl","link","url"):
                    link_col = c; break
            if not link_col:
                for c in df_csv.columns:
                    if df_csv[c].astype(str).str.contains("linkedin.com/in", na=False).any():
                        link_col = c; break
            if not link_col:
                st.error("Non trovo una colonna con i link LinkedIn (es: LinkedIn, linkedinUrl).")
            else:
                urls = (df_csv[link_col].astype(str)
                        .map(normalize_linkedin_url)
                        .dropna()
                        .drop_duplicates()
                        .tolist())
                st.write(f"Trovati {len(urls)} URL unici.")
                if st.button("🚀 Enrich da CSV", key="btn_enrich_csv"):
                    if not APIFY_TOKEN:
                        st.error("Imposta APIFY_TOKEN.")
                    else:
                        apify = ApifyClient(APIFY_TOKEN)
                        with st.spinner(f"Enrichment di {len(urls)} profili…"):
                            enriched = run_linkedin_enrichment(apify, urls)
                        st.session_state["enriched_items"] = enriched
                        df_master = build_people_master(enriched)
                        st.session_state["people_master"] = df_master
                        st.success(f"Creato People Master con {len(df_master)} righe (da CSV).")
                        st.download_button(
                            "⬇️ Scarica CSV (People Master)",
                            df_master.to_csv(index=False).encode("utf-8"),
                            "people_master.csv",
                            "text/csv",
                            key="dl_master_from_csv"
                        )
        except Exception as e:
            st.error(f"Errore CSV: {e}")

# ===== Output finale (se già presente)
if st.session_state.get("people_master") is not None:
    st.markdown("---")
    st.header("👤 People Master (una riga per persona)")
    df_master = st.session_state["people_master"]
    st.dataframe(
        df_master,
        width='stretch',
        height=560,
        column_config={
            "linkedinUrl": st.column_config.LinkColumn("LinkedIn", display_text="Apri profilo"),
            "companyLinkedin": st.column_config.LinkColumn("Company LI", display_text="Azienda"),
            "profilePicHighQuality": st.column_config.LinkColumn("Foto", display_text="Apri foto HQ"),
        }
    )
    st.download_button(
        "⬇️ Scarica CSV (People Master)",
        df_master.to_csv(index=False).encode("utf-8"),
        "people_master.csv",
        "text/csv",
        key="dl_master_bottom"
    )

    # ===== Sezione: Conteggio Università con OpenAI (JSON + Plot) =====
    st.markdown("## 🎓 Conteggio università con OpenAI (JSON + Plot)")
    if not OPENAI_API_KEY:
        st.warning("Imposta OPENAI_API_KEY nella sidebar per usare l'AI.")
    else:
        top_n = st.slider("Top N università da mostrare nel grafico", 5, 50, 15, step=1)
        if st.button("🧠 Conta università (OpenAI)"):
            with st.spinner("Invio dati 'education_top' a OpenAI…"):
                result = count_universities_with_openai(df_master, OPENAI_API_KEY, model=OPENAI_MODEL)

            # 1) Stampa SEMPRE il JSON grezzo
            st.subheader("JSON")
            st.json(result, expanded=True)

            # 2) Plot (se il JSON è valido)
            if isinstance(result, dict) and isinstance(result.get("universities"), dict):
                uni_dict = result["universities"]
                if len(uni_dict) == 0:
                    st.info("Nessuna università riconosciuta dal modello.")
                else:
                    df_plot = (
                        pd.DataFrame(list(uni_dict.items()), columns=["university", "count"])
                        .sort_values("count", ascending=False)
                        .head(top_n)
                    )
                    st.subheader("Bar chart (Top N)")
                    chart = (
                        alt.Chart(df_plot)
                        .mark_bar()
                        .encode(
                            x=alt.X("count:Q", title="Conteggi"),
                            y=alt.Y("university:N", sort='-x', title="Università"),
                            tooltip=["university", "count"]
                        )
                    )
                    st.altair_chart(chart, theme=None)
            else:
                st.error("Risposta del modello non valida per il plotting.")
                
