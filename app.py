"""
Coach - Personal Trainer Web App
==================================
Backend FastAPI que expone un chat con "Coach", tu entrenador personal.
Pensado para desplegarse en un hosting simple (Render, Railway, Fly.io,
tu propia compu, etc.) y usarse desde el navegador del celular.

A diferencia de la version anterior (script de terminal con el Claude
Agent SDK), esta version usa el paquete `anthropic` directo con un loop
de tool-use manual. Es mas portable: no necesita Node.js ni el CLI de
Claude Code instalados en el servidor, solo Python y una ANTHROPIC_API_KEY.

Funcionalidad:
- Genera rutinas personalizadas.
- Registra entrenamientos (ejercicios, series, reps, peso, sensaciones).
- Registra peso corporal y muestra evolucion.
- Recordatorio: si pasaron 3+ dias sin registrar un entrenamiento, el
  banner superior y el propio Coach te lo hacen notar al abrir la app.

Memoria: todo se guarda en archivos JSON locales en ./data (perfil,
entrenamientos, pesos, y la conversacion). Esto es para un solo usuario
(vos). Si lo desplegas en un host con filesystem efimero (se borra en
cada deploy), considera un volumen persistente o una base de datos.
"""

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import anthropic
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1500
MAX_HISTORY_MESSAGES = 40  # acota la conversacion guardada (contexto y costo)
DIAS_PARA_RECORDATORIO = 3

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("COACH_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_FILE = DATA_DIR / "perfil.json"
WORKOUTS_FILE = DATA_DIR / "entrenamientos.json"
WEIGHTS_FILE = DATA_DIR / "pesos.json"
CONVERSATION_FILE = DATA_DIR / "conversacion.json"

# ---------------------------------------------------------------------------
# Almacenamiento
# ---------------------------------------------------------------------------

import json


def _load(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default
    return default


def _save(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Herramientas (funciones Python + su schema para la API)
# ---------------------------------------------------------------------------


def guardar_perfil(
    objetivo: str,
    nivel: str,
    equipamiento: str,
    dias_por_semana: int,
    restricciones: str = "",
) -> str:
    perfil = {
        "objetivo": objetivo,
        "nivel": nivel,
        "equipamiento": equipamiento,
        "dias_por_semana": dias_por_semana,
        "restricciones": restricciones,
        "actualizado": datetime.now().isoformat(timespec="seconds"),
    }
    _save(PROFILE_FILE, perfil)
    return f"Perfil guardado: {json.dumps(perfil, ensure_ascii=False)}"


def leer_perfil() -> str:
    perfil = _load(PROFILE_FILE, None)
    if perfil is None:
        return "No hay perfil guardado todavia."
    return json.dumps(perfil, ensure_ascii=False)


def registrar_entrenamiento(
    ejercicios: str, sensaciones: str = "", fecha: Optional[str] = None
) -> str:
    entrenamientos = _load(WORKOUTS_FILE, [])
    fecha = fecha or date.today().isoformat()
    entrenamientos.append(
        {
            "fecha": fecha,
            "ejercicios": ejercicios,
            "sensaciones": sensaciones,
            "registrado_en": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save(WORKOUTS_FILE, entrenamientos)
    return (
        f"Entrenamiento registrado para {fecha}. "
        f"Total de sesiones registradas: {len(entrenamientos)}."
    )


def leer_historial(limite: int = 5) -> str:
    entrenamientos = _load(WORKOUTS_FILE, [])
    if limite and limite > 0:
        entrenamientos = entrenamientos[-limite:]
    if not entrenamientos:
        return "Todavia no hay entrenamientos registrados."
    return json.dumps(entrenamientos, ensure_ascii=False, indent=2)


def registrar_peso(peso_kg: float, fecha: Optional[str] = None) -> str:
    pesos = _load(WEIGHTS_FILE, [])
    fecha = fecha or date.today().isoformat()
    pesos.append(
        {
            "fecha": fecha,
            "peso_kg": peso_kg,
            "registrado_en": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save(WEIGHTS_FILE, pesos)
    return f"Peso registrado: {peso_kg} kg el {fecha}. Total de registros: {len(pesos)}."


def leer_pesos(limite: int = 10) -> str:
    pesos = _load(WEIGHTS_FILE, [])
    if limite and limite > 0:
        pesos = pesos[-limite:]
    if not pesos:
        return "Todavia no hay registros de peso."
    return json.dumps(pesos, ensure_ascii=False, indent=2)


TOOLS = [
    {
        "name": "guardar_perfil",
        "description": (
            "Guarda o actualiza el perfil del usuario: objetivo, nivel, "
            "equipamiento disponible, dias por semana que puede entrenar y "
            "restricciones o lesiones."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "objetivo": {"type": "string", "description": "Ej: bajar de peso, ganar musculo, resistencia, salud general"},
                "nivel": {"type": "string", "description": "Ej: principiante, intermedio, avanzado"},
                "equipamiento": {"type": "string", "description": "Ej: gimnasio completo, mancuernas en casa, sin equipamiento"},
                "dias_por_semana": {"type": "integer"},
                "restricciones": {"type": "string", "description": "Lesiones o limitaciones. Vacio si no hay."},
            },
            "required": ["objetivo", "nivel", "equipamiento", "dias_por_semana"],
        },
    },
    {
        "name": "leer_perfil",
        "description": "Lee el perfil guardado del usuario. Devuelve vacio si todavia no existe.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "registrar_entrenamiento",
        "description": (
            "Registra una sesion de entrenamiento completada: ejercicios "
            "realizados (con series, repeticiones y peso) y como se sintio "
            "el usuario."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ejercicios": {"type": "string", "description": "Descripcion de ejercicios, series, repeticiones y peso usado."},
                "sensaciones": {"type": "string", "description": "Como se sintio: energia, dolor, dificultad, etc."},
                "fecha": {"type": "string", "description": "Fecha YYYY-MM-DD. Si se omite, se usa hoy."},
            },
            "required": ["ejercicios"],
        },
    },
    {
        "name": "leer_historial",
        "description": "Lee el historial de entrenamientos, limitado a las ultimas N sesiones.",
        "input_schema": {
            "type": "object",
            "properties": {"limite": {"type": "integer", "description": "0 = todas."}},
        },
    },
    {
        "name": "registrar_peso",
        "description": "Registra el peso corporal del usuario en una fecha dada.",
        "input_schema": {
            "type": "object",
            "properties": {
                "peso_kg": {"type": "number"},
                "fecha": {"type": "string", "description": "Fecha YYYY-MM-DD. Si se omite, se usa hoy."},
            },
            "required": ["peso_kg"],
        },
    },
    {
        "name": "leer_pesos",
        "description": "Lee el historial de peso corporal registrado, limitado a los ultimos N registros.",
        "input_schema": {
            "type": "object",
            "properties": {"limite": {"type": "integer", "description": "0 = todos."}},
        },
    },
]

TOOL_FUNCS = {
    "guardar_perfil": guardar_perfil,
    "leer_perfil": leer_perfil,
    "registrar_entrenamiento": registrar_entrenamiento,
    "leer_historial": leer_historial,
    "registrar_peso": registrar_peso,
    "leer_pesos": leer_pesos,
}


def _run_tool(name: str, tool_input: dict) -> str:
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"Herramienta desconocida: {name}"
    try:
        return func(**tool_input)
    except Exception as exc:  # noqa: BLE001 - queremos devolverle el error a Claude, no romper la app
        return f"Error ejecutando {name}: {exc}"


def dias_sin_entrenar() -> Optional[int]:
    entrenamientos = _load(WORKOUTS_FILE, [])
    if not entrenamientos:
        return None
    ultima_fecha = entrenamientos[-1].get("fecha")
    try:
        d = date.fromisoformat(ultima_fecha)
    except (TypeError, ValueError):
        return None
    return (date.today() - d).days


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Sos "Coach", un entrenador personal virtual, cercano y motivador, \
que habla en espanol rioplatense (como en Argentina).

Tu trabajo:
- Conocer al usuario: objetivo, nivel, equipamiento disponible, dias por semana \
disponibles, y restricciones o lesiones. Guarda esta info con guardar_perfil apenas \
la tengas o cuando cambie.
- Al arrancar la conversacion, primero llama a leer_perfil, leer_historial(limite=5) \
y leer_pesos(limite=5) para recordar quien es el usuario, como viene entrenando y su \
evolucion de peso, ANTES de responder. Si no hay perfil, preguntale lo necesario para \
crear uno.
- Generar rutinas personalizadas con ejercicios concretos, series, repeticiones y \
progresion.
- Cuando el usuario cuente que entreno, registrarlo con registrar_entrenamiento.
- Cuando el usuario mencione su peso corporal, registrarlo con registrar_peso.
- Hacer seguimiento real: ajustar la rutina segun el historial, notar rachas o \
inconsistencias, comentar la evolucion de peso si es relevante para el objetivo, y \
motivar sin ser pesado.
- Ser conciso, practico y alentador. No sos medico: ante lesiones serias o dolor \
fuerte, sugeri consultar a un profesional de la salud.

No inventes datos del usuario: si algo no esta en el perfil o el historial, preguntalo."""


def _build_system_prompt() -> str:
    dias = dias_sin_entrenar()
    if dias is not None and dias >= DIAS_PARA_RECORDATORIO:
        return (
            SYSTEM_PROMPT
            + f"\n\nNOTA: pasaron {dias} dias desde el ultimo entrenamiento "
            "registrado. Si esto es el arranque de la conversacion, mencionaselo "
            "al usuario de forma motivadora (sin retarlo) y pregunta como viene."
        )
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="Coach - Personal Trainer")

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno
    return _client


class ChatIn(BaseModel):
    message: str


@app.post("/chat")
def chat(body: ChatIn):
    client = get_client()
    history = _load(CONVERSATION_FILE, [])
    history.append({"role": "user", "content": body.message})
    messages = history[-MAX_HISTORY_MESSAGES:]
    system = _build_system_prompt()

    assistant_content = []
    for _ in range(6):  # tope de vueltas del loop de tools por mensaje
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        assistant_content = resp.content
        messages.append(
            {
                "role": "assistant",
                "content": [block.model_dump() for block in assistant_content],
            }
        )

        if resp.stop_reason != "tool_use":
            break

        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                result_text = _run_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    final_text = "".join(
        block.text for block in assistant_content if block.type == "text"
    )

    _save(CONVERSATION_FILE, messages)

    return {"reply": final_text, "dias_sin_entrenar": dias_sin_entrenar()}


@app.get("/estado")
def estado():
    return {"dias_sin_entrenar": dias_sin_entrenar()}


@app.get("/health")
def health():
    return {"ok": True}


# Sirve el frontend estatico (tiene que ir al final: monta "/" como catch-all)
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")
