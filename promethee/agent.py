"""The brain: LLM loop (Ollama) that calls MCP tools until it has an answer to speak.

The agent itself speaks French: the system prompt, the tool descriptions and everything the
model reads or says stay in French. Only the code around it is in English.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime

import ollama

from .config import Settings
from .mcp_hub import McpHub
from .metrics import Turn

WEEKDAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

BASE_PROMPT = """Tu es Prométhée, un assistant vocal personnel et généraliste, comme un ami cultivé et serviable. Tu discutes de tout : culture générale, conseils, idées, blagues, calculs, actualité, bourse, sport, sorties, cinéma, météo… Tu as aussi des outils pour agir sur certaines données de la maison, mais ce n'est qu'une partie de ce que tu sais faire : ne réponds jamais que tu ne t'occupes que d'un sujet.
Tes réponses sont lues à voix haute par une synthèse vocale.

Façon de parler :
- Réponds en français, en une à trois phrases courtes et naturelles, sur un ton chaleureux. Pour une simple discussion, réponds directement, sans outil.
- Jamais de markdown, de listes à puces, d'emojis, de tableaux ni d'adresses web : uniquement du texte qui se dit bien.
- Écris les dates, heures et nombres comme on les dit (« le 15 novembre », « 388 euros 10 »).

Internet :
- Dès qu'une question porte sur quelque chose de récent, qui change souvent ou de local (actualité, cours de bourse, météo, séances de cinéma, horaires, prix, résultats sportifs, adresses), utilise web_search au lieu de répondre de mémoire. Si tu n'es pas sûr d'un fait, cherche-le aussi.
- Si les extraits ne contiennent pas la réponse précise, ouvre le résultat le plus prometteur avec read_page, puis un autre si besoin.
- Résume l'essentiel : pour une liste (films, titres de l'actualité…), cite trois ou quatre éléments au plus et propose d'en dire plus. Ne cite pas les sites. Si tu ne trouves pas, dis-le simplement.{location}

Outils de la maison :
- Utilise les outils dès qu'une question porte sur des données qu'ils gèrent ; n'invente jamais leur contenu.
- Les données changent entre deux demandes : appelle de nouveau l'outil à chaque fois, même si tu as déjà répondu à une question proche.
- Ne dis jamais que tu vérifies, ajoutes ou supprimes quelque chose sans avoir réellement appelé l'outil correspondant.
- N'invente jamais un identifiant : reprends-le tel quel d'un résultat d'outil, sinon rappelle l'outil de recherche pour l'obtenir. Ne le prononce jamais.
- Enchaîne toi-même tous les outils nécessaires avant de répondre : par exemple chercher l'article pour obtenir son identifiant, puis le supprimer. Ne rends la main qu'une fois la demande entièrement faite.
- N'annonce jamais une action (« je vais supprimer… ») : fais-la avec l'outil, puis dis ce qui a été fait.
- Avant de supprimer ou de modifier : si la recherche trouve plusieurs articles et que la demande ne précise pas lequel (date, quantité, nom exact), ne touche à rien et demande lequel en citant les choix. Ne supprime plusieurs articles que si on te le demande explicitement (« tous », « les deux »).
- Si une demande est ambiguë ou qu'il manque une information, pose une question courte au lieu d'agir.
Nous sommes le {date}, il est {time}."""

# Read last by the model: without this reminder, a tool's instructions ("you manage the inventory")
# end up making it believe that's all it can do.
REMINDER = """Rappel : tu restes un assistant généraliste. Les consignes ci-dessus ne valent que pour les demandes qui concernent ces outils ; pour tout le reste, discute librement ou cherche sur Internet."""

STALE_TOOL_RESULT = "(résultat consulté à ce moment-là, à redemander pour des données à jour)"
# Tool results of the previous turn, as kept in the history: enough for the IDs of a house tool
# ("yes, delete it"), while three pages read on the web no longer fill the context of the next turn.
HISTORY_TOOL_RESULT_CHARS = 1500
# Same idea for the last-chance round, which must fit even when the context overflowed.
RESCUE_TOOL_RESULT_CHARS = 1200
RESCUE_MAX_TOKENS = 200
RESCUE_PROMPT = ("Réponds maintenant à la dernière demande, en une à trois phrases courtes, avec ce que "
                 "les résultats ci-dessus contiennent. S'ils ne suffisent pas, dis simplement ce que tu "
                 "as trouvé et ce qui manque.")

# Spoken fallbacks
UNFINISHED_ANSWER = "Désolé, je n'ai pas réussi à finir cette demande."
EMPTY_ANSWER = "Je n'ai pas de réponse."


def clean_for_speech(text: str) -> str:
    # Some "thinking" models omit the opening tag: keep what follows the last closing one.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<think>.*", "", text, flags=re.S)  # unfinished thinking
    text = re.sub(r"[*_#`>|]", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " […]"


class Agent:
    def __init__(self, settings: Settings, hub: McpHub):
        self.s = settings
        self.hub = hub
        self.client = ollama.AsyncClient(host=settings.ollama_host)
        # One item per turn: the request, the tool calls (results trimmed) and the answer.
        self.history: list[list] = []
        self.last_turn_at = 0.0

    def _system_prompt(self) -> str:
        now = datetime.now()
        location = ""
        if self.s.home_location:
            location = (f"\n- L'utilisateur habite à {self.s.home_location} : pour une question locale "
                        "(« à côté », « près de chez moi », la météo), ajoute ce lieu à la recherche.")
        prompt = BASE_PROMPT.format(
            date=f"{WEEKDAYS_FR[now.weekday()]} {now:%d/%m/%Y}", time=f"{now:%H:%M}", location=location,
        )
        for name, instructions in self.hub.instructions.items():
            prompt += (f"\n\nConsignes de l'outil « {name} », uniquement pour les demandes qui le "
                       f"concernent :\n{instructions}")
        if self.hub.instructions:
            prompt += f"\n\n{REMINDER}"
        return prompt

    async def warmup(self) -> None:
        """Loads the model into memory before the first turn (otherwise the first turn pays for it)."""
        await self.client.chat(
            model=self.s.llm_model, messages=[{"role": "user", "content": "ok"}],
            think=False, options={**self.s.llm_options(), "num_predict": 1}, keep_alive="1h",
        )

    @staticmethod
    def _compact(turn_msgs: list, answer: str) -> list:
        """Keeps tool calls in the history, otherwise the model ends up imitating answers given
        without tools and makes things up. Results are kept in full here: `_past` decides which
        ones stay visible."""
        kept: list = []
        for m in turn_msgs:
            role = m["role"] if isinstance(m, dict) else m.role
            if role == "assistant" and not m.tool_calls:
                continue  # raw final answer: replaced by the cleaned-up version
            if role == "tool":
                kept.append({**m, "content": _trim(m["content"], HISTORY_TOOL_RESULT_CHARS)})
            elif role == "assistant":
                kept.append({"role": "assistant", "content": m.content or "", "tool_calls": m.tool_calls})
            else:
                kept.append(m)
        return [*kept, {"role": "assistant", "content": answer}]

    def _past(self) -> list:
        """History sent to the model. Tool results from the last turn stay visible (trimmed by
        `_compact`): a confirmation ("yes, delete it") needs the IDs they contain. Older ones are
        replaced by a placeholder: they would bloat the context and be stale."""
        past: list = []
        for i, t in enumerate(self.history):
            last = i == len(self.history) - 1
            for m in t:
                if m.get("role") == "tool" and not last:
                    m = {**m, "content": STALE_TOOL_RESULT}
                past.append(m)
        return past

    async def ask(self, text: str, turn: Turn) -> str:
        if time.monotonic() - self.last_turn_at > self.s.history_ttl_s:
            self.history = []  # new conversation
        past = self._past()
        messages: list = [{"role": "system", "content": self._system_prompt()}, *past,
                          {"role": "user", "content": text}]
        start = len(messages) - 1  # start of this turn in `messages`
        answer = ""
        finished = False
        for _ in range(self.s.llm_max_rounds):
            resp = await self._chat(messages, turn, tools=self.hub.ollama_tools or None,
                                    think=self.s.llm_think)
            msg = resp.message
            messages.append(msg)
            if not msg.tool_calls:
                answer = clean_for_speech(msg.content or "")
                finished = True
                break
            for call in msg.tool_calls:
                name, args = call.function.name, dict(call.function.arguments or {})
                turn.tool_calls.append(name)
                print(f"  🔧 {name}({json.dumps(args, ensure_ascii=False)})")
                with turn.step("tools"):
                    result = await self.hub.call(name, args)
                messages.append({"role": "tool", "content": result, "tool_name": name})

        if not answer:
            # Empty answer (the context filled up, or the thinking used the room left) or out of
            # rounds: one last short round to answer with whatever the tools brought back.
            reason = "réponse vide" if finished else "plus de passes"
            print(f"  🛟 Passe de secours ({reason})")
            answer = await self._rescue(messages[start:], turn)
        if not answer:
            answer = EMPTY_ANSWER if finished else UNFINISHED_ANSWER
        self.history = [*self.history, self._compact(messages[start:], answer)][-self.s.history_turns:]
        self.last_turn_at = time.monotonic()
        return answer

    async def _chat(self, messages: list, turn: Turn, *, tools, think: bool, options: dict | None = None):
        with turn.step("llm"):
            resp = await self.client.chat(
                model=self.s.llm_model, messages=messages, tools=tools, think=think,
                options={**self.s.llm_options(), **(options or {})}, keep_alive="1h",
            )
        turn.llm_rounds += 1
        turn.prompt_tokens += resp.prompt_eval_count or 0
        turn.prompt_seconds += (resp.prompt_eval_duration or 0) / 1e9
        turn.gen_tokens += resp.eval_count or 0
        turn.gen_seconds += (resp.eval_duration or 0) / 1e9
        prompt = resp.prompt_eval_count or 0
        if resp.done_reason == "length" or prompt >= self.s.llm_ctx * 0.95:
            print(f"  ⚠️  Contexte plein : prompt {prompt} tokens sur {self.s.llm_ctx} "
                  f"(fin : {resp.done_reason}). Augmente LLM_CTX si ça se répète.")
        return resp

    async def _rescue(self, turn_msgs: list, turn: Turn) -> str:
        """Last-chance round: only this turn (no history), tool results trimmed, no tools and no
        thinking, so that the room left goes to the answer."""
        messages: list = [{"role": "system", "content": self._system_prompt()}]
        for m in turn_msgs:
            role = m["role"] if isinstance(m, dict) else m.role
            if role == "tool":
                messages.append({**m, "content": _trim(m["content"], RESCUE_TOOL_RESULT_CHARS)})
            elif role == "assistant":
                if m.tool_calls:  # the final empty answer, if any, is dropped
                    messages.append({"role": "assistant", "content": "", "tool_calls": m.tool_calls})
            else:
                messages.append(m)
        messages.append({"role": "user", "content": RESCUE_PROMPT})
        resp = await self._chat(messages, turn, tools=None, think=False,
                                options={"num_predict": RESCUE_MAX_TOKENS})
        return clean_for_speech(resp.message.content or "")
