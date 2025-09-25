# streamlit run app.py
# Requisiti:
#   pip install streamlit apify-client openai pandas python-dotenv
# ENV (opzionali): APIFY_TOKEN, OPENAI_API_KEY

import os
import json
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple, Set

import pandas as pd
import streamlit as st
from apify_client import ApifyClient
from openai import OpenAI

# ----------------------------
# Configurazione pagina
# ----------------------------
st.set_page_config(
    page_title="Google SERP → JSON Viewer (Apify + OpenAI)",
    layout="wide",
)

# ----------------------------
# Helpers comuni
# ----------------------------
def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def download_dataset_items(client: ApifyClient, dataset_id: str) -> List[Dict[str, Any]]:
    """Scarica tutti gli item dal dataset di Apify (lista di pagine SERP)."""
    items = []
    for it in client.dataset(dataset_id).iterate_items():
        items.append(it)
    return items

def save_json(items: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def items_to_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """Appiattisce le organicResults in una tabella comoda."""
    if not items:
        return pd.DataFrame()
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

# ----------------------------
# Estrazione locale (no-AI)
# ----------------------------
def normalize_linkedin_url(u: str) -> str:
    """Normalizza URL LinkedIn per dedup (hostname lower, rimuove query/fragment, no slash finale)."""
    try:
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

def split_name_from_title(title: str) -> Tuple[str, str]:
    """
    Estrae Nome e Cognome da titoli del tipo:
    "Nome Cognome - Stealth Startup" | "Nome Cognome | Stealth Mode" | "Nome Cognome — ...".
    Heuristic semplice ma efficace nella maggior parte dei casi.
    """
    if not title:
        return "", ""
    first_part = re.split(r"\s[-–—|·•]\s", title, maxsplit=1)[0]
    first_part = re.sub(r"\s+\([^)]*\)$", "", first_part).strip()
    first_part = re.sub(r"^\s*LinkedIn\s*›\s*", "", first_part, flags=re.IGNORECASE).strip()
    tokens = [t for t in re.split(r"\s+", first_part) if t]
    if len(tokens) >= 2:
        return tokens[0], " ".join(tokens[1:])
    return first_part, ""

def extract_people_from_serp(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """Ritorna DataFrame [Nome, Cognome, LinkedIn] usando solo parsing locale del JSON Apify."""
    rows = []
    for page in items:
        for r in page.get("organicResults", []):
            url = r.get("url") or r.get("link") or r.get("sourceUrl")
            if not url or "linkedin.com" not in url:
                continue
            # preferisci profili /in/ o /pub/; accetta altri solo se non chiari
            if not re.search(r"linkedin\.com/(in|pub)/", url):
                # alcuni profili localizzati possono avere formati diversi: non escludere a priori
                pass
            title = r.get("title") or r.get("displayedUrl") or ""
            nome, cognome = split_name_from_title(title)
            rows.append({"Nome": nome, "Cognome": cognome, "LinkedIn": url})

    # dedup su LinkedIn normalizzato
    seen: Set[str] = set()
    dedup = []
    for row in rows:
        norm = normalize_linkedin_url(row["LinkedIn"])
        if norm not in seen:
            seen.add(norm)
            row["LinkedIn"] = norm
            dedup.append(row)
    return pd.DataFrame(dedup, columns=["Nome", "Cognome", "LinkedIn"])

def build_people_table(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Crea una tabella 'people' direttamente dagli organicResults:
    Nome | Cognome | Title | Snippet | Location | Followers | LinkedIn
    """
    rows = []
    for page in items:
        for r in page.get("organicResults", []):
            url = r.get("url") or r.get("link") or r.get("sourceUrl")
            if not url or "linkedin.com" not in url:
                continue

            # Prendiamo i campi richiesti
            title = r.get("title") or ""
            snippet = r.get("description") or r.get("snippet") or ""
            location = (r.get("personalInfo") or {}).get("location")
            followers = r.get("followersAmount")

            # Nome/Cognome dal title (heuristic)
            nome, cognome = split_name_from_title(title)
            url_norm = normalize_linkedin_url(url)

            rows.append({
                "Nome": nome,
                "Cognome": cognome,
                "Title": title,
                "Snippet": snippet,
                "Location": location,
                "Followers": followers,
                "LinkedIn": url_norm,  # sarà reso cliccabile nella UI
            })

    df = pd.DataFrame(rows, columns=["Nome", "Cognome", "Title", "Snippet", "Location", "Followers", "LinkedIn"])

    # Dedup per link
    if not df.empty:
        df = df.drop_duplicates(subset=["LinkedIn"]).reset_index(drop=True)
    return df

# ----------------------------
# Funzioni AI (OpenAI)
# ----------------------------
def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def run_openai_batch_read(client_oa: OpenAI, items: List[Dict[str, Any]], model: str = "gpt-4o-mini", batch_size: int = 50) -> Tuple[List[str], str]:
    """Crea executive summary + URL LinkedIn via AI, leggendo il JSON a batch."""
    all_urls: Set[str] = set()
    partial_summaries: List[str] = []
    batches = chunk_list(items, batch_size)

    for idx, batch in enumerate(batches, start=1):
        minimal_batch = [
            {
                "title": it.get("title"),
                "url": it.get("url") or it.get("link") or it.get("sourceUrl"),
                "snippet": it.get("snippet") or it.get("description"),
                "displayedUrl": it.get("displayedUrl"),
                "searchQuery": it.get("searchQuery"),
            }
            for it in batch
        ]

        prompt = (
            "Sei un assistente che analizza risultati Google in formato JSON.\n"
            "Dal seguente batch di risultati estrai:\n"
            "1) Tutte le URL di profili/domini LinkedIn trovate (lista deduplicata);\n"
            "2) Nomi di città/località italiane citate (se presenti);\n"
            "3) 5-10 parole chiave ricorrenti;\n"
            "4) Un riassunto di 3-5 righe del contenuto del batch.\n\n"
            "Rispondi in JSON con le chiavi: urls, cities, keywords, summary.\n\n"
            f"Batch size: {len(minimal_batch)}\n"
        )

        response = client_oa.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Rispondi solo in JSON valido."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": json.dumps(minimal_batch, ensure_ascii=False)},
            ],
            temperature=0.2,
        )

        content = getattr(response, "output_text", None) or response.choices[0].message.content
        try:
            data = json.loads(content)
        except Exception:
            try:
                start = content.find("{"); end = content.rfind("}")
                data = json.loads(content[start:end+1])
            except Exception:
                data = {"urls": [], "cities": [], "keywords": [], "summary": content}

        for u in data.get("urls", []) or []:
            all_urls.add(u)
        partial_summaries.append(f"[Batch {idx}/{len(batches)}] " + (data.get("summary") or ""))

    final_prompt = (
        "Unisci i seguenti riassunti parziali in un unico executive summary (max 250 parole) "
        "e proponi 5 azioni utili. Rispondi in italiano.\n\n"
        + "\n\n".join(partial_summaries)
    )
    final_resp = client_oa.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "Sei un analista conciso."},
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.3,
    )
    final_summary = getattr(final_resp, "output_text", None) or final_resp.choices[0].message.content
    return sorted(all_urls), final_summary

def analyze_json_people_ai(client_oa: OpenAI, items: List[Dict[str, Any]], model: str = "gpt-4o-mini", batch_size: int = 60) -> pd.DataFrame:
    """Pulsante ANALIZZA JSON (AI): estrae Nome/Cognome/LinkedIn con prompt richiesto."""
    def _minimalize(batch):
        return [
            {
                "title": it.get("title"),
                "url": it.get("url") or it.get("link") or it.get("sourceUrl"),
                "snippet": it.get("snippet") or it.get("description"),
            }
            for it in batch
        ]

    all_rows = []
    batches = chunk_list(items, batch_size)
    for batch in batches:
        minimal = _minimalize(batch)
        user_prompt = (
            "leggi il file json e dammi Nome cognome link linkedin\n\n"
            "Regole:\n"
            "- Considera solo risultati che sembrano profili LinkedIn personali (non aziende/pagine scuola).\n"
            "- Se non sei sicuro del nome/cognome, prova a inferirlo dal title/snippet; altrimenti lascia vuoto.\n"
            "- Rispondi SOLO con un JSON valido: una lista di oggetti con chiavi esattamente: "
            '"Nome", "Cognome", "LinkedIn".\n'
            "- Niente testo fuori dal JSON.\n"
        )
        resp = client_oa.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Sei un estrattore di dati: restituisci solo JSON valido (lista di oggetti)."},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": json.dumps(minimal, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
        content = getattr(resp, "output_text", None) or resp.choices[0].message.content
        try:
            data_list = json.loads(content)
        except Exception:
            try:
                start = content.find("["); end = content.rfind("]")
                data_list = json.loads(content[start:end+1])
            except Exception:
                data_list = []

        for row in data_list:
            nome = (row.get("Nome") or "").strip()
            cognome = (row.get("Cognome") or "").strip()
            url = (row.get("LinkedIn") or "").strip()
            if url and "linkedin.com" in url:
                all_rows.append({"Nome": nome, "Cognome": cognome, "LinkedIn": url})

    # Dedup su LinkedIn normalizzato
    seen = set(); dedup_rows = []
    for r in all_rows:
        norm = normalize_linkedin_url(r["LinkedIn"])
        if norm not in seen:
            seen.add(norm)
            dedup_rows.append({"Nome": r["Nome"], "Cognome": r["Cognome"], "LinkedIn": norm})
    return pd.DataFrame(dedup_rows, columns=["Nome", "Cognome", "LinkedIn"])

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("🔐 Credenziali")
APIFY_TOKEN = st.sidebar.text_input("APIFY_TOKEN", get_env("APIFY_TOKEN", ""), type="password")
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY (per AI)", get_env("OPENAI_API_KEY", ""), type="password")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Impostazioni Actor")
default_query = 'site:linkedin.com/in ("stealth mode" OR "stealth startup" OR "in stealth") (Italy OR Italia OR Milano OR Roma OR Torino OR Firenze OR Bologna)'
query_text = st.sidebar.text_area("Query Google", value=default_query, height=100)
country_code = st.sidebar.text_input("countryCode", value="it")
max_pages = st.sidebar.number_input("maxPagesPerQuery (~10 risultati per pagina)", min_value=1, max_value=50, value=10, step=1)
site_filter = st.sidebar.text_input("site", value="www.linkedin.com")
ai_mode = st.sidebar.selectbox("aiMode", ["aiModeOff", "aiModeBrief", "aiModeFull"], index=0)
save_html_to_kv = st.sidebar.checkbox("saveHtmlToKeyValueStore", value=True)
include_icons = st.sidebar.checkbox("includeIcons", value=False)
mobile_results = st.sidebar.checkbox("mobileResults", value=False)

st.sidebar.markdown("---")
st.sidebar.header("🧪 OpenAI (opzioni)")
use_ai = st.sidebar.checkbox("Usa OpenAI (summary/estrazioni)", value=True)
ai_model = st.sidebar.text_input("Modello OpenAI", value="gpt-4o-mini")
batch_size = st.sidebar.number_input("Batch size (item/chiamata)", min_value=20, max_value=200, value=50, step=10)

# ----------------------------
# Header
# ----------------------------
st.title("🔎 Google SERP → JSON Viewer (Apify + OpenAI)")
st.caption("Apify `apify/google-search-scraper` → Dataset → JSON → Tabella • Estrazione locale + AI (summary & parsing)")

# ----------------------------
# Tabs
# ----------------------------
tab_run, tab_upload = st.tabs(["▶️ Esegui scraper", "📤 Carica JSON esistente"])

# Stato condiviso
if "items" not in st.session_state:
    st.session_state["items"] = []
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
if "urls_ai" not in st.session_state:
    st.session_state["urls_ai"] = []
if "summary_ai" not in st.session_state:
    st.session_state["summary_ai"] = ""
if "people_df_local" not in st.session_state:
    st.session_state["people_df_local"] = pd.DataFrame(columns=["Nome", "Cognome", "LinkedIn"])
if "people_df_ai" not in st.session_state:
    st.session_state["people_df_ai"] = pd.DataFrame(columns=["Nome", "Cognome", "LinkedIn"])

# ----------------------------
# TAB 1: Esegui scraper
# ----------------------------
with tab_run:
    st.subheader("1) Avvio Actor Apify")
    st.write("**Nota**: Google ora mostra ~10 risultati organici per pagina; `resultsPerPage` è ignorato → usa `maxPagesPerQuery`.")
    colA, colB = st.columns([1, 2])

    with colA:
        start_btn = st.button("🚀 Avvia scraping e scarica dataset")

    with colB:
        st.json({
            "aiMode": ai_mode,
            "countryCode": country_code,
            "focusOnPaidAds": False,
            "forceExactMatch": False,
            "includeIcons": include_icons,
            "includeUnfilteredResults": False,
            "maxPagesPerQuery": int(max_pages),
            "maximumLeadsEnrichmentRecords": 0,
            "mobileResults": mobile_results,
            "queries": query_text,
            "resultsPerPage": 100,  # Ignorato da Google
            "saveHtml": False,
            "saveHtmlToKeyValueStore": save_html_to_kv,
            "site": site_filter,
        })

    if start_btn:
        if not APIFY_TOKEN:
            st.error("Imposta APIFY_TOKEN nella sidebar.")
        else:
            with st.spinner("Esecuzione actor su Apify..."):
                apify = ApifyClient(APIFY_TOKEN)
                run_input = {
                    "aiMode": ai_mode,
                    "countryCode": country_code,
                    "focusOnPaidAds": False,
                    "forceExactMatch": False,
                    "includeIcons": include_icons,
                    "includeUnfilteredResults": False,
                    "maxPagesPerQuery": int(max_pages),
                    "maximumLeadsEnrichmentRecords": 0,
                    "mobileResults": mobile_results,
                    "queries": query_text,
                    "resultsPerPage": 100,  # Ignorato
                    "saveHtml": False,
                    "saveHtmlToKeyValueStore": save_html_to_kv,
                    "site": site_filter,
                }

                run = apify.actor("apify/google-search-scraper").call(run_input=run_input)
                dataset_id = run["defaultDatasetId"]

                items = download_dataset_items(apify, dataset_id)
                st.session_state["items"] = items
                st.session_state["df"] = items_to_dataframe(items)

                out_path = "google_serp_results.json"
                save_json(items, out_path)
                st.success(f"Scaricate {len(items)} pagine SERP. Salvate in `{out_path}`.")

                st.download_button(
                    label="⬇️ Scarica JSON",
                    data=json.dumps(items, ensure_ascii=False, indent=2),
                    file_name="google_serp_results.json",
                    mime="application/json",
                )

    if len(st.session_state["items"]) > 0:
        st.markdown("---")
        st.subheader("📋 People (Locale) — Nome, Cognome, Title, Snippet, Location, Followers, LINK")

        df_people_table = build_people_table(st.session_state["items"])

        if df_people_table.empty:
            st.info("Nessun profilo LinkedIn rilevato negli organicResults.")
        else:
            # Mostra tabella con link cliccabile
            st.dataframe(
                df_people_table,
                use_container_width=True,
                height=500,
                column_config={
                    "LinkedIn": st.column_config.LinkColumn(
                        "LINK",
                        help="Apri il profilo su LinkedIn",
                        display_text="Apri profilo"
                    )
                },
                hide_index=True,
            )

            # Download CSV della tabella people
            st.download_button(
                label="⬇️ Scarica CSV (People)",
                data=df_people_table.to_csv(index=False).encode("utf-8"),
                file_name="people_table.csv",
                mime="text/csv",
            )


# ----------------------------
# TAB 2: Carica JSON esistente
# ----------------------------
with tab_upload:
    st.subheader("Carica un file JSON già scaricato")
    up = st.file_uploader("Seleziona `google_serp_results.json`", type=["json"])
    if up is not None:
        try:
            items = json.load(up)
            if not isinstance(items, list):
                st.error("Il JSON deve essere una lista (array).")
            else:
                st.session_state["items"] = items
                st.session_state["df"] = items_to_dataframe(items)
                st.success(f"Caricate {len(items)} pagine SERP dal file.")
        except Exception as e:
            st.error(f"Errore nel parsing JSON: {e}")

    if len(st.session_state["items"]) > 0:
        st.dataframe(st.session_state["df"], use_container_width=True, height=500)
        st.download_button(
            label="⬇️ Riscaria JSON (normalizzato)",
            data=json.dumps(st.session_state["items"], ensure_ascii=False, indent=2),
            file_name="google_serp_results.json",
            mime="application/json",
        )

# ----------------------------
# Sezione Estrazioni & AI
# ----------------------------
if len(st.session_state["items"]) > 0:
    st.markdown("---")
    st.header("🔧 Estrazioni & 🤖 AI")

    colL, colR = st.columns(2)

    # ------- Lato sinistro: locale ------
    with colL:
        st.subheader("📊 ANALIZZA JSON (Locale)")
        st.caption("Parsing locale del JSON per Nome / Cognome / LinkedIn (senza AI)")

        if st.button("📊 Esegui analisi locale"):
            df_local = extract_people_from_serp(st.session_state["items"])
            st.session_state["people_df_local"] = df_local
            st.success(f"Estratti (locale) {len(df_local)} profili.")

        if not st.session_state["people_df_local"].empty:
            st.dataframe(st.session_state["people_df_local"], use_container_width=True, height=350)
            csv_bytes = st.session_state["people_df_local"].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Scarica CSV (Locale)", csv_bytes, "people_linkedin_local.csv", "text/csv")

    # ------- Lato destro: AI ------
    with colR:
        st.subheader("🤖 ANALIZZA JSON (AI)")
        st.caption("Prompt: **leggi il file json e dammi Nome cognome link linkedin** + Executive Summary")

        if use_ai and OPENAI_API_KEY:
            client_oa = OpenAI(api_key=OPENAI_API_KEY)

            c1, c2 = st.columns(2)
            with c1:
                run_ai_summary = st.button("🧠 Executive summary + URL (AI)")
            with c2:
                run_ai_people = st.button("📊 Estrai Nome/Cognome/LinkedIn (AI)")

            if run_ai_summary:
                try:
                    with st.spinner("AI: creo executive summary e raccolgo URL..."):
                        urls_ai, summary_ai = run_openai_batch_read(
                            client_oa=client_oa,
                            items=st.session_state["df"].to_dict("records"),  # usa il DF appiattito per token-efficienza
                            model=ai_model,
                            batch_size=int(batch_size),
                        )
                        st.session_state["urls_ai"] = urls_ai
                        st.session_state["summary_ai"] = summary_ai
                    st.success(f"AI OK. URL LinkedIn deduplicate: {len(urls_ai)}")
                except Exception as e:
                    st.error(f"Errore AI (summary/urls): {e}")

            if run_ai_people:
                try:
                    with st.spinner("AI: estraggo Nome/Cognome/LinkedIn..."):
                        df_people_ai = analyze_json_people_ai(
                            client_oa=client_oa,
                            items=st.session_state["df"].to_dict("records"),
                            model=ai_model,
                            batch_size=int(batch_size),
                        )
                        st.session_state["people_df_ai"] = df_people_ai
                    st.success(f"AI OK. Profili estratti: {len(st.session_state['people_df_ai'])}")
                except Exception as e:
                    st.error(f"Errore AI (people): {e}")

            # risultati AI
            if st.session_state["urls_ai"]:
                st.subheader("🔗 URL LinkedIn (AI)")
                st.dataframe(pd.DataFrame({"url": st.session_state["urls_ai"]}), use_container_width=True, height=250)
                st.download_button(
                    "⬇️ Scarica URL LinkedIn (AI .txt)",
                    "\n".join(st.session_state["urls_ai"]),
                    "linkedin_urls_ai.txt",
                    "text/plain",
                )

            if st.session_state["summary_ai"]:
                st.subheader("📝 Executive summary (AI)")
                st.write(st.session_state["summary_ai"])

            if not st.session_state["people_df_ai"].empty:
                st.subheader("📋 People (AI) — Nome / Cognome / LinkedIn")
                st.dataframe(st.session_state["people_df_ai"], use_container_width=True, height=350)
                csv_ai = st.session_state["people_df_ai"].to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Scarica CSV (AI)", csv_ai, "people_linkedin_ai.csv", "text/csv")

        else:
            st.warning("Per usare l’AI attiva la spunta 'Usa OpenAI' e imposta OPENAI_API_KEY nella sidebar.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "Suggerimenti: aumenta `maxPagesPerQuery` per più risultati (~10 per pagina). "
    "`resultsPerPage` è ignorato da Google; lasciato per compatibilità."
)
