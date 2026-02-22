#!/usr/bin/env python3
"""
Kore — Ejemplo rápido
═════════════════════

    python3 examples/quickstart.py

IAFiscal © 2026
"""

from kore import Kore, DomainConfig

# ── 1. Crear motor con dominio fiscal ─────────────────────────────────────
config = DomainConfig(
    name="fiscal",
    categories={
        "iva": ["iva", "soportado", "repercutido", "factura", "tipo impositivo", "21%", "10%", "4%"],
        "nomina": ["nomina", "salario", "seguridad social", "cotizacion", "retencion"],
        "contabilidad": ["asiento", "cuenta", "debe", "haber", "balance", "amortizacion", "cierre"],
    },
    min_score=0.4,
)

k = Kore(domain="demo_fiscal", db_path=":memory:", config=config)

# ── 2. Ingestar conocimiento ──────────────────────────────────────────────
datos = [
    {"query": "¿Cuáles son los tipos de IVA en España?",
     "answer": "Los tipos de IVA en España son: general 21%, reducido 10%, superreducido 4%."},
    {"query": "¿Cómo se genera un asiento de nómina?",
     "answer": "Debe: 640 Sueldos y salarios. Haber: 476 Organismos SS, 4751 HP retencion IRPF, 465 Remuneraciones pendientes."},
    {"query": "¿Qué es el IVA soportado?",
     "answer": "El IVA soportado es el IVA que paga una empresa al comprar bienes o servicios. Se registra en la cuenta 472."},
    {"query": "¿Cómo se contabiliza un leasing?",
     "answer": "El leasing se contabiliza según la NRV 8ª del PGC como arrendamiento financiero, activando el bien y registrando la deuda."},
    {"query": "¿Cuándo se presenta el modelo 303?",
     "answer": "El modelo 303 se presenta trimestralmente: 1T en abril, 2T en julio, 3T en octubre, 4T en enero del año siguiente."},
    {"query": "¿Qué es la amortización?",
     "answer": "La amortización es el reflejo contable de la depreciación de un activo. Se calcula según tablas oficiales y se registra en cuentas 68x/28x."},
    {"query": "¿Qué retención de IRPF se aplica en nómina?",
     "answer": "La retención depende de la situación personal y el salario. Los tramos van del 19% al 47% sobre la base liquidable."},
    {"query": "¿Qué es una factura rectificativa?",
     "answer": "Una factura rectificativa corrige errores o modificaciones de una factura anterior. Puede ser por devolución, descuento o error en datos."},
]

result = k.ingest(datos)
print(f"Ingestadas {result['loaded']} entries\n")

# ── 3. Preguntar ──────────────────────────────────────────────────────────
queries = [
    "tipos de IVA en España",           # → hit exacto (normalizado)
    "como hago un asiento de nomina",    # → hit keywords o BM25
    "contabilizar un leasing",           # → hit BM25
    "receta de tortilla de patatas",     # → miss (no es fiscal)
    "amortizacion inmovilizado",         # → hit keywords/BM25
]

print("=" * 65)
print("  CONSULTAS")
print("=" * 65)

for q in queries:
    r = k.ask(q)
    if r:
        print(f"\n  [{r.level.value:8s} {r.score:.2f}]  {q}")
        print(f"  → {r.answer[:100]}...")
    else:
        print(f"\n  [MISS         ]  {q}")

# ── 4. Feedback (aprendizaje) ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  APRENDIZAJE")
print("=" * 65)

# El usuario valida una respuesta nueva
k.feedback(
    "¿Cómo se hace el cierre del ejercicio?",
    "El cierre se realiza con asientos de regularización (cuentas 6/7→129), cierre de balance (activo/pasivo) y apertura del nuevo ejercicio.",
    positive=True,
)

# Ahora Kore ya sabe responder
r = k.ask("cierre del ejercicio")
if r:
    print(f"\n  Aprendido: [{r.level.value} {r.score:.2f}]")
    print(f"  → {r.answer[:120]}...")

# ── 5. Stats ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ESTADÍSTICAS")
print("=" * 65)

import json
print(json.dumps(k.stats().to_dict(), indent=2, ensure_ascii=False))

k.close()
