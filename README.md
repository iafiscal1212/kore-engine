<p align="center">
  <h1 align="center">Kore Engine</h1>
  <p align="center"><strong>El motor de conocimiento que responde antes que tu LLM.</strong></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/dependencias-0-brightgreen" alt="Zero deps">
  <img src="https://img.shields.io/badge/latencia-%3C1ms-orange" alt="<1ms">
  <img src="https://img.shields.io/badge/licencia-MIT-lightgrey" alt="MIT">
</p>

---

## El problema

Cada empresa que usa un LLM gasta miles de euros al mes respondiendo las mismas preguntas una y otra vez. El 70-80% de las consultas de tus usuarios ya las has respondido antes.

## La solución

```
pip install kore-engine
```

Kore se pone **delante de tu LLM**. Si sabe la respuesta, la da en menos de 1 milisegundo. Si no sabe, te dice que no sabe y tú decides si pasarla al LLM. Cada respuesta buena que pasa por Kore se queda para la próxima vez.

```python
from kore import Kore

k = Kore("mi_dominio")
k.ingest("mi_conocimiento.jsonl")

r = k.ask("¿cómo se calcula el IVA?")
if r and r.confident:
    return r.answer          # <1ms, 0 tokens, 0 coste
else:
    return llamar_al_llm()   # Solo cuando Kore no sabe
```

## Por qué Kore

| | Kore | RAG + VectorDB | LLM directo |
|---|---|---|---|
| **Instalación** | `pip install kore-engine` | Redis + Pinecone + embeddings + orquestador | API key |
| **Dependencias** | 0 | 4-6 servicios | 1 API |
| **Latencia** | <1ms | 50-200ms | 500-3000ms |
| **Coste por query** | 0 | ~$0.001 (embeddings) | $0.01-0.10 |
| **Aprende solo** | Si | No | No |
| **Memoria conversacional** | Si | No | Limitada |

## Qué puede hacer

**Responder sin LLM** — Kore tiene múltiples niveles de comprensión. No es un simple cache de texto exacto. Entiende variaciones, sinónimos y contexto.

**Recordar conversaciones** — Si un usuario pregunta "tipos de IVA" y luego dice "y el repercutido?", Kore sabe que habla de IVA repercutido.

```python
k.ask("tipos de IVA en España", session_id="user_1")
# → "General 21%, reducido 10%, superreducido 4%"

k.ask("y el repercutido?", session_id="user_1")
# → "El IVA repercutido es el que cobra la empresa al vender..."
```

**Aprender de las interacciones** — Cuando tu LLM da una buena respuesta y el usuario la valida, Kore la incorpora. La próxima vez no necesita LLM.

```python
k.feedback("¿cómo se cierra el ejercicio?", respuesta_del_llm, positive=True)
# A partir de ahora, Kore responde esa pregunta solo
```

**Cualquier dominio** — Fiscal, legal, médico, soporte técnico, e-commerce, educación. Tú defines las categorías de tu dominio y Kore se adapta.

```python
k = Kore("soporte_tecnico", config=DomainConfig(
    name="soporte",
    categories={
        "hardware": ["pc", "monitor", "teclado", "ratón", "impresora"],
        "software": ["windows", "office", "email", "vpn", "antivirus"],
        "red": ["wifi", "ethernet", "dns", "proxy", "firewall"],
    },
))
```

**Ingestar de todo** — JSONL, CSV, JSON, ChatML, conversations. Le metes tu conocimiento en el formato que tengas.

```python
k.ingest("faq.jsonl")
k.ingest("soporte.csv")
k.ingest([{"query": "...", "answer": "..."}])
```

**Estadísticas en tiempo real** — Sabes exactamente cuánto te ahorras.

```python
s = k.stats()
# → hit_rate: 78%, queries: 15420, hits: 12028, misses: 3392
```

## Comprensión semántica (opcional)

El paquete base ya es potente. Pero si quieres comprensión por significado (no solo por palabras), activa el nivel semántico:

```
pip install kore-engine[semantic]
```

```python
k = Kore("mi_dominio", semantic=True)
```

Añade ~35MB. Sin PyTorch (2GB+). Sin GPU. Corre en cualquier CPU.

## Rendimiento real

Probado con un dataset de **7,888 pares pregunta-respuesta** de contabilidad española:

| Métrica | Resultado |
|---|---|
| Ingesta | 7,888 entries en 1.1s |
| Hit rate en queries del dominio | **100%** |
| Falsos positivos (queries fuera de dominio) | **0%** |
| Latencia media | **0.9ms/query** |
| Memoria conversacional | 16/16 queries contextuales resueltas |
| Sesiones paralelas | Aislamiento perfecto |

## Persistencia

```python
# En memoria (rápido, volátil)
k = Kore("dominio", db_path=":memory:")

# En disco (persiste entre reinicios)
k = Kore("dominio", db_path="mi_conocimiento.db")
```

## API completa

```python
from kore import Kore, DomainConfig

k = Kore("dominio", config=config)

# ── Conocimiento
k.ingest(source)                          # Carga masiva
k.add(query, answer)                      # Entrada individual
k.clear()                                 # Limpiar todo

# ── Consulta
r = k.ask(query)                          # Sin sesión
r = k.ask(query, session_id="user_1")     # Con memoria

# ── Resultado
r.answer                                  # Respuesta
r.score                                   # Confianza (0.0 - 1.0)
r.confident                               # True si score >= 0.7
r.level                                   # Nivel de matching
r.category                                # Categoría detectada

# ── Memoria
k.context(session_id)                     # Estado de la sesión
k.history(session_id)                     # Últimos turnos
k.forget(session_id)                      # Olvidar sesión

# ── Aprendizaje
k.feedback(query, answer, positive=True)  # Registrar feedback
k.learn()                                 # Incorporar feedback acumulado

# ── Estadísticas
k.stats()                                 # Hit rate, contadores, etc.
```

## Requisitos

- Python 3.10+
- Nada más

## Licencia

MIT — [IAFiscal](https://iafiscal.es) © 2026
