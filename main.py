import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiohttp
import fitz  # pymupdf
import httpx
import openai
from nexus_client import NexusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
# Authentik OAuth (client credentials) — remplace l'ancienne clé API statique
# MQTT_SERVICE_API_KEY, rejetée par mosquitto-auth-authentik qui exige un vrai JWT
# Authentik (voir _new_nexus_client ci-dessous, même pattern qu'agent-news/wiki-agent).
AUTHENTIK_URL = os.environ.get("AUTHENTIK_URL", "https://sso.caronboulme.fr")
AUTHENTIK_CLIENT_ID = os.environ["AUTHENTIK_CLIENT_ID"]
AUTHENTIK_CLIENT_SECRET = os.environ["AUTHENTIK_CLIENT_SECRET"]
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://thebrain:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-4-12B-it-qat-w4a16-ct")


async def _new_nexus_client() -> NexusClient:
    """Obtient un token OAuth Client Credentials via Authentik et construit un
    NexusClient authentifié — mosquitto-auth-authentik valide le mot de passe MQTT
    comme un JWT Authentik, un token frais est donc nécessaire à chaque connexion."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{AUTHENTIK_URL}/application/o/token/",
            data={
                "grant_type": "client_credentials",
                "client_id": AUTHENTIK_CLIENT_ID,
                "client_secret": AUTHENTIK_CLIENT_SECRET,
            },
        )
        if resp.status_code != 200:
            logger.error(f"Échec obtention token OAuth: {resp.status_code} {resp.text}")
            raise RuntimeError("Cannot get OAuth token")
        access_token = resp.json()["access_token"]

    return await NexusClient.from_authentik_token(
        AUTHENTIK_URL, MQTT_HOST, access_token,
        AUTHENTIK_CLIENT_ID, AUTHENTIK_CLIENT_SECRET, MQTT_PORT,
    )
LLAMACPP_API_KEY = os.environ["LLAMACPP_API_KEY"]

CACHE_FILE = os.environ.get("CACHE_FILE", "/data/cache.json")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/data/chroma")
PDF_FOLDER = os.environ.get("PDF_FOLDER", "/pdfs")
DOC_COUNTER_INITIAL = int(os.environ.get("DOC_COUNTER_INITIAL", "30"))
COUNTER_DECREMENT_INTERVAL = int(os.environ.get("COUNTER_DECREMENT_INTERVAL", str(24 * 3600)))
FOLDER_RESCAN_INTERVAL = int(os.environ.get("FOLDER_RESCAN_INTERVAL", "300"))

AGENT_NAME = "pdf"
_subscribed_sessions: set[str] = set()

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
# En dessous de ce nombre de caractères extraits, une page est considérée "scannée"
# (image sans couche texte) et bascule sur la transcription vision de thebrain.
MIN_TEXT_PER_PAGE = int(os.environ.get("MIN_TEXT_PER_PAGE", "20"))
# Garde-fou : un PDF scanné de centaines de pages ne doit pas déclencher des centaines
# d'appels vision (coût/latence) — au-delà, les pages restantes sont simplement ignorées.
MAX_VISION_PAGES = int(os.environ.get("MAX_VISION_PAGES", "20"))

DETAIL_SYSTEM_PROMPTS = {
    1: (
        "Réponds à la question en UNE seule phrase directe et naturelle, à l'oral. "
        "Utilise uniquement les extraits de document fournis. Pas de citation de pages."
    ),
    2: (
        "Rédige un paragraphe clair et factuel répondant à la question, basé uniquement "
        "sur les extraits de document fournis. 3 à 5 phrases. Français, ton oral."
    ),
    3: (
        "Rédige une réponse détaillée avec tous les faits pertinents des extraits fournis. "
        "Mentionne la page si c'est utile à l'oral (ex: 'à la page 3...'). Français, ton oral."
    ),
}


# ---------------------------------------------------------------------------
# Document cache — même mécanique "compteur" que CityCache (agent-weather) :
# chaque consultation remet le compteur à DOC_COUNTER_INITIAL (30 jours), une
# décrémentation quotidienne purge (cache + chunks Chroma) les documents inactifs.
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_since(iso: str) -> float:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return float("inf")


class DocCache:
    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path) as f:
                    self._data = json.load(f)
                logger.info(f"Cache chargé: {len(self._data)} document(s)")
        except Exception as e:
            logger.error(f"Erreur chargement cache: {e}")
            self._data = {}

    def _save_sync(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erreur sauvegarde cache: {e}")

    def is_indexed(self, doc_id: str, fingerprint: str) -> bool:
        entry = self._data.get(doc_id, {})
        return bool(entry.get("chunk_ids")) and entry.get("fingerprint") == fingerprint

    def get_chunk_ids(self, doc_id: str) -> list[str]:
        return self._data.get(doc_id, {}).get("chunk_ids", [])

    def last_decremented(self, doc_id: str) -> str | None:
        return self._data.get(doc_id, {}).get("last_decremented")

    def active_docs(self) -> list[str]:
        return [k for k, v in self._data.items() if v.get("counter", 0) > 0]

    async def upsert(self, doc_id: str, kind: str, source: str, chunk_ids: list[str], fingerprint: str):
        async with self._lock:
            entry = self._data.get(doc_id, {})
            entry.update({
                "kind": kind,
                "source": source,
                "chunk_ids": chunk_ids,
                "fingerprint": fingerprint,
                "counter": DOC_COUNTER_INITIAL,
            })
            if "last_decremented" not in entry:
                entry["last_decremented"] = _now_iso()
            self._data[doc_id] = entry
            self._save_sync()
        logger.info(f"Cache upsert: {doc_id} ({len(chunk_ids)} chunk(s)), compteur remis à {DOC_COUNTER_INITIAL}")

    async def touch(self, doc_id: str):
        async with self._lock:
            if doc_id in self._data:
                self._data[doc_id]["counter"] = DOC_COUNTER_INITIAL
                self._save_sync()

    async def decrement_all(self, collection):
        async with self._lock:
            expired = [k for k, v in self._data.items() if v.get("counter", 1) - 1 <= 0]
            for doc_id in expired:
                chunk_ids = self._data[doc_id].get("chunk_ids", [])
                if chunk_ids:
                    try:
                        collection.delete(ids=chunk_ids)
                    except Exception as e:
                        logger.error(f"Suppression Chroma échouée pour {doc_id}: {e}")
                logger.info(f"Document expiré du cache: {doc_id}")
                del self._data[doc_id]
            for entry in self._data.values():
                entry["counter"] = entry.get("counter", 1) - 1
                entry["last_decremented"] = _now_iso()
            self._save_sync()
        logger.info(f"Compteurs décrémentés — {len(self._data)} document(s) actifs")


# ---------------------------------------------------------------------------
# ChromaDB — collection unique "pdf_documents", un point par chunk de texte.
# ---------------------------------------------------------------------------

_chroma_client = None
_collection = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name="pdf_documents",
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            ),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Extraction PDF (texte natif + fallback vision pour les pages scannées)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, doc_id: str, page_number: int) -> list[tuple[str, dict]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 1 > CHUNK_SIZE:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:] + "\n" + para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)
    return [
        (chunk, {"doc_id": doc_id, "page": page_number, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]


def _page_image_data_url(page: "fitz.Page") -> str:
    import base64
    pix = page.get_pixmap(dpi=150)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    return f"data:image/png;base64,{b64}"


def _vision_transcribe_sync(data_url: str) -> str:
    client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Transcris tout le texte visible sur cette page de "
                                          "document, tel quel, sans commentaire ni résumé."},
            ],
        }],
        max_tokens=2000,
    )
    return resp.choices[0].message.content or ""


def _extract_pdf_sync(pdf_bytes: bytes, doc_id: str) -> list[tuple[str, dict]]:
    """Bloquant — à appeler via run_in_executor. Texte natif par page, fallback vision
    (thebrain, format image_url déjà utilisé côté Joshua) si peu/pas de texte extractible."""
    chunks: list[tuple[str, dict]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    vision_pages_used = 0
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text().strip()
            if len(text) < MIN_TEXT_PER_PAGE:
                if vision_pages_used >= MAX_VISION_PAGES:
                    logger.warning(f"[{doc_id}] page {page_index + 1} scannée, limite vision atteinte, ignorée")
                    continue
                try:
                    text = _vision_transcribe_sync(_page_image_data_url(page))
                    vision_pages_used += 1
                    logger.info(f"[{doc_id}] page {page_index + 1} transcrite via vision ({len(text)} chars)")
                except Exception as e:
                    logger.error(f"[{doc_id}] échec vision page {page_index + 1}: {e}")
                    continue
            if not text:
                continue
            chunks.extend(_chunk_text(text, doc_id, page_index + 1))
    finally:
        doc.close()
    return chunks


async def index_document(doc_id: str, pdf_bytes: bytes, cache: DocCache) -> list[str]:
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, _extract_pdf_sync, pdf_bytes, doc_id)
    collection = _get_collection()
    old_ids = cache.get_chunk_ids(doc_id)
    if old_ids:
        try:
            collection.delete(ids=old_ids)
        except Exception as e:
            logger.warning(f"[{doc_id}] purge des anciens chunks échouée: {e}")
    ids = [hashlib.md5(f"{doc_id}::{i}".encode()).hexdigest() for i in range(len(chunks))]
    if chunks:
        collection.add(ids=ids, documents=[c[0] for c in chunks], metadatas=[c[1] for c in chunks])
    logger.info(f"[{doc_id}] indexé: {len(chunks)} chunk(s)")
    return ids


# ---------------------------------------------------------------------------
# Sources : dossier partagé (lecture seule côté contenu, écriture uniquement
# pour "save") et URL (téléchargement à la volée, cache 30 jours par défaut).
# ---------------------------------------------------------------------------

async def _fetch_pdf_url(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                logger.warning(f"Téléchargement {url}: HTTP {resp.status}")
                return None
            ct = resp.headers.get("Content-Type", "")
            if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
                logger.warning(f"Content-Type inattendu pour {url}: {ct!r}")
            return await resp.read()
    except Exception as e:
        logger.error(f"Téléchargement échoué {url}: {e}")
        return None


def _safe_filename_from_url(url: str) -> str:
    name = unquote(os.path.basename(urlparse(url).path)) or "document"
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def _save_pdf_to_folder_sync(pdf_bytes: bytes, url: str) -> str:
    folder = Path(PDF_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    base_name = _safe_filename_from_url(url)
    stem, ext = os.path.splitext(base_name)
    candidate = base_name
    i = 1
    while (folder / candidate).exists():
        candidate = f"{stem}_{i}{ext}"
        i += 1
    (folder / candidate).write_bytes(pdf_bytes)
    logger.info(f"Document enregistré: {candidate}")
    return candidate


# ---------------------------------------------------------------------------
# Boucles de fond
# ---------------------------------------------------------------------------

async def folder_scan_loop(cache: DocCache):
    while True:
        try:
            folder = Path(PDF_FOLDER)
            if folder.is_dir():
                for path in sorted(folder.rglob("*.pdf")):
                    doc_id = str(path.relative_to(folder))
                    fingerprint = f"{path.stat().st_mtime_ns}:{path.stat().st_size}"
                    if cache.is_indexed(doc_id, fingerprint):
                        continue
                    logger.info(f"Nouveau/modifié dans {PDF_FOLDER}: {doc_id}")
                    try:
                        chunk_ids = await index_document(doc_id, path.read_bytes(), cache)
                        await cache.upsert(doc_id, "file", doc_id, chunk_ids, fingerprint)
                    except Exception as e:
                        logger.error(f"Indexation échouée {doc_id}: {e}")
        except Exception as e:
            logger.error(f"Erreur scan dossier: {e}")
        await asyncio.sleep(FOLDER_RESCAN_INTERVAL)


async def counter_decrement_loop(cache: DocCache):
    collection = _get_collection()
    for doc_id in cache.active_docs():
        last = cache.last_decremented(doc_id)
        if last:
            days_missed = int(_seconds_since(last) // COUNTER_DECREMENT_INTERVAL)
            if days_missed > 0:
                logger.info(f"Rattrapage {days_missed} décrémentation(s) manquée(s)")
                for _ in range(days_missed):
                    await cache.decrement_all(collection)
                break

    while True:
        await asyncio.sleep(COUNTER_DECREMENT_INTERVAL)
        await cache.decrement_all(collection)


# ---------------------------------------------------------------------------
# Synthèse LLM (thebrain) — complétion simple, sans tool-calling forcé : constaté en
# test que thebrain (gemma4, --tool-call-parser=gemma4) renvoie parfois
# finish_reason="tool_calls" avec message.tool_calls=None sur des contextes réels
# (échec silencieux du parseur), alors que la même requête sans "tools"/tool_choice
# produit une réponse en texte libre cohérente à chaque fois.
# ---------------------------------------------------------------------------

def _synthesize_sync(query: str, chunks: list[dict], detail_level: int) -> str:
    system_prompt = DETAIL_SYSTEM_PROMPTS.get(detail_level, DETAIL_SYSTEM_PROMPTS[2])
    context_text = "\n\n".join(
        f"[{c['metadata'].get('doc_id')}, page {c['metadata'].get('page')}]\n{c['content']}"
        for c in chunks
    )
    user_content = f"Question: {query}\n\nExtraits du/des document(s):\n{context_text}"
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=600,
        )
        report = (resp.choices[0].message.content or "").strip()
        return report or context_text[:500]
    except Exception as e:
        logger.error(f"Synthèse échouée: {e}")
        return context_text[:500]


# ---------------------------------------------------------------------------
# Traitement d'une requête
# ---------------------------------------------------------------------------

async def _index_url_document(document: str, cache: DocCache) -> bytes | None:
    """Indexe le document s'il ne l'est pas déjà (fingerprint = l'URL elle-même : on ne
    re-télécharge/ré-indexe pas tant que l'utilisateur ne le redemande pas après
    expiration). Retourne les bytes du PDF si un téléchargement a eu lieu (utile pour
    'save'), sinon None (déjà en cache)."""
    if cache.is_indexed(document, document):
        await cache.touch(document)
        return None
    async with aiohttp.ClientSession() as session:
        pdf_bytes = await _fetch_pdf_url(session, document)
    if pdf_bytes is None:
        return None
    chunk_ids = await index_document(document, pdf_bytes, cache)
    await cache.upsert(document, "url", document, chunk_ids, fingerprint=document)
    return pdf_bytes


async def _handle_pdf_request(
    query: str, document: str | None, save: bool, detail_level: int, cache: DocCache,
) -> tuple[str, list[dict], str | None]:
    doc_id = None
    saved_as = None

    if document:
        is_url = document.startswith("http://") or document.startswith("https://")
        if is_url:
            doc_id = document
            pdf_bytes = await _index_url_document(document, cache)
            if not cache.is_indexed(document, document):
                return "Je n'ai pas pu télécharger ce document.", [], None
            if save:
                # 'save' explicitement demandé : on écrit le PDF dans le dossier partagé.
                # Le fichier sera ensuite indexé en permanence par folder_scan_loop, en plus
                # de l'entrée URL (qui elle continue d'expirer normalement après 30 jours
                # d'inactivité — pas de fusion des deux entrées, complexité non justifiée).
                if pdf_bytes is None:
                    async with aiohttp.ClientSession() as session:
                        pdf_bytes = await _fetch_pdf_url(session, document)
                if pdf_bytes is not None:
                    saved_as = await asyncio.get_event_loop().run_in_executor(
                        None, _save_pdf_to_folder_sync, pdf_bytes, document
                    )
        else:
            doc_id = document
            path = Path(PDF_FOLDER) / doc_id
            if not path.is_file():
                return "Je n'ai pas trouvé ce document dans les documents disponibles.", [], None
            fingerprint = f"{path.stat().st_mtime_ns}:{path.stat().st_size}"
            if cache.is_indexed(doc_id, fingerprint):
                await cache.touch(doc_id)
            else:
                chunk_ids = await index_document(doc_id, path.read_bytes(), cache)
                await cache.upsert(doc_id, "file", doc_id, chunk_ids, fingerprint)

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return "Aucun document n'est disponible pour l'instant.", [], saved_as

    where = {"doc_id": doc_id} if doc_id else None
    raw = collection.query(query_texts=[query], n_results=min(6, count), where=where)
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    if not docs:
        return "Je n'ai rien trouvé de pertinent dans les documents disponibles.", [], saved_as

    chunks = [{"content": d, "metadata": m} for d, m in zip(docs, metas)]

    # Toute consultation (même via recherche large, pas seulement un accès direct par nom/URL)
    # prolonge le TTL du document concerné.
    for found_doc_id in {m.get("doc_id") for m in metas if m.get("doc_id")}:
        if cache.get_chunk_ids(found_doc_id):
            await cache.touch(found_doc_id)

    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(None, _synthesize_sync, query, chunks, detail_level)

    sources, seen = [], set()
    for m in metas:
        key = (m.get("doc_id"), m.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"document": m.get("doc_id"), "page": m.get("page")})

    return report, sources, saved_as


# ---------------------------------------------------------------------------
# MQTT — même flow que agent-weather/agent-web-search
# ---------------------------------------------------------------------------

async def on_user_connected(topic: str, payload, cache: DocCache):
    if not isinstance(payload, dict):
        return

    username = payload.get("username")
    authenticated = payload.get("authenticated")
    session_id = payload.get("session_id")
    private_topics = payload.get("private_topics", [])

    if not username or not authenticated or not session_id:
        return

    agent_topics_topic = None
    for agent_entry in private_topics:
        for t in agent_entry.get("topics", []):
            if t["topic"].endswith("/agent_topics"):
                agent_topics_topic = t["topic"]
                break

    if not agent_topics_topic:
        logger.warning(f"[{username}] agent_topics topic introuvable, skip")
        return

    request_topic = f"users/{username}/{session_id}/pdf/request"
    result_topic = f"users/{username}/{session_id}/pdf/result"

    nexus = await _new_nexus_client()

    await nexus.publish(
        agent_topics_topic,
        [{
            "agent": AGENT_NAME,
            "topics": [
                {
                    "topic": request_topic,
                    "description": (
                        "Répond à une question sur un document PDF (rapport, manuel, facture, "
                        "notice...) que l'utilisateur a dans ses documents, ou dont il donne "
                        "l'URL. 'query' est OBLIGATOIRE. 'document' : IMPORTANT — dès que "
                        "l'utilisateur donne ou a précédemment donné une URL vers un PDF, ou "
                        "cite un nom de fichier précis (ex: 'facture_edf.pdf'), tu DOIS "
                        "reporter cette URL/ce nom tel quel dans 'document', même si ce n'est "
                        "pas la première question sur ce document dans la conversation — sans "
                        "quoi la recherche se fait sur TOUS les documents disponibles et peut "
                        "mélanger plusieurs documents entre eux. Ne laisse 'document' vide que "
                        "si l'utilisateur n'a donné ni URL ni nom de fichier précis. 'save' "
                        "(uniquement utile avec une URL) : à mettre à true SEULEMENT si "
                        "l'utilisateur demande explicitement de garder/enregistrer/sauvegarder "
                        "ce document — sinon un document reçu par URL n'est conservé que 30 jours."
                    ),
                    "access": "write",
                    "response_topic": result_topic,
                    "format": {
                        "query": "Que dit ce document sur ...",
                        "document": "facture_edf.pdf",
                        "save": False,
                        "detail_level": 2,
                    },
                },
                {
                    "topic": result_topic,
                    "description": (
                        "Réponse basée sur le contenu du/des document(s). Utiliser le champ "
                        "'report' pour répondre. Si 'saved_as' est non-nul, confirmer à "
                        "l'utilisateur que le document a été enregistré sous ce nom."
                    ),
                    "access": "read",
                    "format": {
                        "report": "string",
                        "sources": [{"document": "string", "page": 1}],
                        "saved_as": None,
                    },
                },
            ],
        }],
        retain=True,
    )
    logger.info(f"[{username}/{session_id}] Topics déclarés sur {agent_topics_topic}")

    if session_id in _subscribed_sessions:
        logger.debug(f"[{username}/{session_id}] Déjà abonné, skip")
        return

    _subscribed_sessions.add(session_id)

    async def on_pdf_request(t: str, p):
        if not isinstance(p, dict):
            logger.warning(f"[{username}/{session_id}] Payload PDF non-dict, ignoré: {p!r}")
            await nexus.publish(result_topic, {
                "report": "Erreur: requête mal formée.",
                "sources": [], "saved_as": None,
                "error": "malformed request: expected a JSON object",
            })
            return

        try:
            query = (p.get("query") or "").strip()
            if not query:
                logger.warning(f"[{username}/{session_id}] Requête sans 'query', ignorée: {p}")
                await nexus.publish(result_topic, {
                    "report": "Erreur: aucune question fournie.", "sources": [], "saved_as": None,
                })
                return

            document = p.get("document") or None
            save = bool(p.get("save", False))
            detail_level = int(p.get("detail_level", 2))
            if detail_level not in (1, 2, 3):
                detail_level = 2

            logger.info(f"[{username}] Requête pdf: {query!r} document={document!r} save={save}")

            report, sources, saved_as = await _handle_pdf_request(query, document, save, detail_level, cache)
        except Exception as e:
            logger.error(f"[{username}/{session_id}] Échec traitement requête pdf: {e}")
            await nexus.publish(result_topic, {
                "report": "Erreur: une erreur interne est survenue lors du traitement de la requête.",
                "sources": [], "saved_as": None,
                "error": f"internal error while handling request: {e}",
            })
            return

        await nexus.publish(result_topic, {"report": report, "sources": sources, "saved_as": saved_as})
        logger.info(f"[{username}/{session_id}] Réponse publiée sur {result_topic}")

    nexus.subscribe(request_topic, on_pdf_request)
    nexus.start_listening()
    logger.info(f"[{username}/{session_id}] Abonné à {request_topic}")


async def main():
    cache = DocCache(CACHE_FILE)

    asyncio.create_task(folder_scan_loop(cache))
    asyncio.create_task(counter_decrement_loop(cache))

    nexus = await _new_nexus_client()

    async def _on_user_connected(t: str, p):
        await on_user_connected(t, p, cache)

    nexus.subscribe("common/user_connected", _on_user_connected)
    nexus.start_listening()
    logger.info(f"PDF service démarré — dossier: {PDF_FOLDER}, cache: {CACHE_FILE}, compteur initial: {DOC_COUNTER_INITIAL}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
