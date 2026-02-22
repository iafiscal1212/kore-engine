# Kore Engine

**Motor de Conocimiento Inteligente** — Cache + comprensión + decisión en un solo paquete.

```
pip install kore-engine
```

Cero dependencias. Un import. Funciona.

## Qué hace

Kore es la capa inteligente que va **delante de cualquier LLM**. Si sabe la respuesta, la da en <1ms sin gastar un token. Si no sabe, dice "no sé" y dejas pasar la query al LLM.

```python
from kore import Kore

k = Kore("mi_dominio")
k.ingest("conocimiento.jsonl")

r = k.ask("¿cómo se calcula el IVA?")
if r:
    print(r.answer)       # respuesta inmediata
    print(r.score)        # 0.0 - 1.0
    print(r.confident)    # True si score >= 0.7
else:
    # pasar al LLM
    ...
```

## 4 niveles de retrieval

| Nivel | Método | Velocidad | Dependencias |
|-------|--------|-----------|--------------|
| 1 | Hash exacto | <0.1ms | Ninguna |
| 2 | Keywords + categoría | <1ms | Ninguna |
| 3 | BM25 ranking | <5ms | Ninguna |
| 4 | Semántico ONNX | <15ms | `pip install kore-engine[semantic]` (~35MB) |

Los niveles 1-3 son **Python puro**. El nivel 4 es opcional.

## Ingesta de conocimiento

```python
# Desde archivo JSONL
k.ingest("datos.jsonl")

# Desde CSV
k.ingest("datos.csv")

# Desde lista de dicts
k.ingest([
    {"query": "¿Qué es X?", "answer": "X es..."},
    {"query": "¿Cómo se hace Y?", "answer": "Se hace..."},
])

# Entrada individual
k.add("¿Qué es Z?", "Z es esto")
```

Formatos JSONL soportados: `query/answer`, `conversations`, ChatML.

## Dominios configurables

```python
from kore import Kore, DomainConfig

config = DomainConfig(
    name="fiscal",
    categories={
        "iva": ["iva", "soportado", "repercutido", "factura"],
        "nomina": ["nomina", "salario", "seguridad social"],
    },
    min_score=0.4,
)

k = Kore("fiscal", config=config)
```

## Aprendizaje

Kore aprende de las interacciones validadas por los usuarios:

```python
# El usuario valida una respuesta del LLM
k.feedback("¿Cómo se hace X?", "Se hace así...", positive=True)

# La próxima vez, Kore responde sin LLM
r = k.ask("¿Cómo se hace X?")  # → hit
```

## Estadísticas

```python
s = k.stats()
print(s.hit_rate)       # 0.78
print(s.total_entries)  # 7888
print(s.to_dict())      # JSON completo
```

## Nivel semántico (opcional)

Para activar comprensión semántica real (similitud por significado, no solo por palabras):

```
pip install kore-engine[semantic]
```

```python
k = Kore("mi_dominio", semantic=True)
```

Usa un mini-transformer ONNX (~35MB total) que corre en CPU. Sin PyTorch, sin GPU.

## Licencia

MIT — IAFiscal © 2026
