#!/usr/bin/env python3
"""
Kore — Test de Memoria Conversacional con CONTABILIZA-ES
════════════════════════════════════════════════════════════
Simula conversaciones reales de asesores fiscales donde
las preguntas de seguimiento son vagas y dependen del contexto.

IAFiscal © 2026
"""

import time
import json
import tempfile
import requests
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kore import Kore, DomainConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN = "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"
DATASET_REPO = "Carmenest/contabilidad-es-dataset"

FISCAL_CONFIG = DomainConfig(
    name="fiscal",
    categories={
        "iva": [
            "iva", "impuesto valor", "soportado", "repercutido", "deducible",
            "exento", "intracomunitaria", "factura rectificativa", "tipo impositivo",
            "base imponible", "cuota", "prorrata", "regimen especial",
            "modelo 303", "modelo 390", "sii", "21%", "10%", "4%",
        ],
        "irpf": [
            "irpf", "retencion", "retenciones", "renta", "pagos fraccionados",
            "modelo 111", "modelo 190", "modelo 100", "rendimiento",
            "deduccion", "minimo personal", "tramo",
        ],
        "sociedades": [
            "impuesto sociedades", "modelo 200", "modelo 202", "resultado contable",
            "base imponible negativa", "tipo gravamen", "pago fraccionado",
        ],
        "nomina": [
            "nomina", "salario", "sueldo", "seguridad social", "cotizacion",
            "contingencias", "desempleo", "formacion", "fogasa",
            "convenio", "finiquito", "indemnizacion", "paga extra",
        ],
        "contabilidad": [
            "asiento", "cuenta", "debe", "haber", "balance", "pgc",
            "plan general contable", "cierre", "apertura", "regularizacion",
            "amortizacion", "deterioro", "provision", "periodificacion",
            "existencias", "inmovilizado", "leasing",
        ],
        "facturacion": [
            "factura", "facturacion", "albaran", "presupuesto", "cobro",
            "pago", "proveedor", "cliente", "vencimiento", "recibo",
            "verifactu", "factura electronica",
        ],
        "mercantil": [
            "cuentas anuales", "deposito", "registro mercantil", "junta",
            "administrador", "capital social", "reserva", "dividendo",
        ],
        "laboral": [
            "contrato", "alta", "baja", "siltra", "sistema red",
            "accidente", "incapacidad", "maternidad", "despido", "ere", "erte",
        ],
        "boicac": [
            "boicac", "icac", "consulta", "resolucion", "norma registro",
            "neca", "memoria",
        ],
    },
    min_score=0.4,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONVERSACIONES SIMULADAS
# ─────────────────────────────────────────────────────────────────────────────

CONVERSACIONES = [
    {
        "titulo": "Asesor pregunta sobre IVA → seguimiento vago",
        "session_id": "asesor_maria",
        "turnos": [
            ("¿Cuáles son los tipos de IVA en España?", None),
            ("y el soportado?", "Debería responder sobre IVA soportado"),
            ("y el repercutido?", "Debería responder sobre IVA repercutido"),
            ("cómo se hace?", "Debería explicar el asiento o procedimiento del IVA"),
        ],
    },
    {
        "titulo": "Autónomo pregunta sobre nóminas → cambia a IVA",
        "session_id": "autonomo_pedro",
        "turnos": [
            ("¿Cómo se contabiliza una nómina?", None),
            ("y la seguridad social?", "Debería responder sobre SS en contexto de nómina"),
            ("ahora necesito saber sobre IVA intracomunitario", None),
            ("y la factura?", "Debería responder sobre factura en contexto de IVA"),
        ],
    },
    {
        "titulo": "Consulta sobre contabilidad → profundiza",
        "session_id": "contable_luis",
        "turnos": [
            ("¿Cómo se contabiliza un leasing?", None),
            ("y la amortización?", "Debería responder sobre amortización en contexto de leasing/inmov."),
            ("explica más", "Debería ampliar sobre amortización"),
            ("y el cierre del ejercicio?", None),
        ],
    },
    {
        "titulo": "Dos usuarios en paralelo NO se mezclan",
        "session_id": None,  # Se gestionan manualmente
        "turnos": [],  # Se ejecuta aparte
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA + INGESTA
# ─────────────────────────────────────────────────────────────────────────────

def cargar_dataset(k: Kore) -> int:
    """Descarga CONTABILIZA-ES y lo ingesta en Kore."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    all_lines = []
    seen = set()

    for filename in ["dataset_completo.jsonl", "train.jsonl"]:
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{filename}"
        print(f"  Descargando {filename}...", end=" ", flush=True)
        try:
            resp = requests.get(url, headers=headers, timeout=120)
            resp.raise_for_status()
            lines = resp.text.strip().split('\n')
            for line in lines:
                if line not in seen:
                    all_lines.append(line)
                    seen.add(line)
            print(f"{len(lines)} lineas")
        except Exception as e:
            print(f"ERROR: {e}")

    tmp = Path(tempfile.mktemp(suffix='.jsonl'))
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines))

    t0 = time.perf_counter()
    result = k.ingest(tmp)
    elapsed = time.perf_counter() - t0
    print(f"  Ingestadas {result['loaded']} entries en {elapsed:.1f}s")
    tmp.unlink(missing_ok=True)
    return result['loaded']


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_conversacion(k: Kore, conv: dict):
    """Ejecuta una conversación simulada y muestra resultados."""
    sid = conv["session_id"]
    print(f"\n{'─' * 65}")
    print(f"  {conv['titulo']}")
    print(f"  session_id: {sid}")
    print(f"{'─' * 65}")

    for i, (query, expectativa) in enumerate(conv["turnos"], 1):
        t0 = time.perf_counter()
        r = k.ask(query, session_id=sid)
        elapsed = (time.perf_counter() - t0) * 1000

        # Obtener contexto de sesión
        ctx = k.context(sid)

        if r:
            preview = r.answer[:120].replace("\n", " ")
            print(f"\n  [{i}] Usuario: \"{query}\"")
            print(f"      Kore:    [{r.level.value:8s} {r.score:.2f}] {preview}...")
            print(f"      Contexto: topic={ctx['topic']}, entities={ctx['entities'][:5]}, {elapsed:.1f}ms")
            if expectativa:
                print(f"      Esperado: {expectativa}")
        else:
            print(f"\n  [{i}] Usuario: \"{query}\"")
            print(f"      Kore:    [MISS] — sin respuesta")
            print(f"      Contexto: topic={ctx['topic']}, entities={ctx['entities'][:5]}, {elapsed:.1f}ms")


def test_sesiones_paralelas(k: Kore):
    """Verifica que dos sesiones no se mezclan."""
    print(f"\n{'─' * 65}")
    print(f"  Dos usuarios en paralelo NO se mezclan")
    print(f"{'─' * 65}")

    # User A habla de IVA
    r_a1 = k.ask("tipos de IVA en España", session_id="user_A")
    # User B habla de nóminas
    r_b1 = k.ask("asiento de nómina con seguridad social", session_id="user_B")

    ctx_a = k.context("user_A")
    ctx_b = k.context("user_B")

    print(f"\n  User A pregunta: \"tipos de IVA\"")
    print(f"    → topic={ctx_a['topic']}, entities={ctx_a['entities'][:5]}")
    print(f"  User B pregunta: \"asiento de nómina\"")
    print(f"    → topic={ctx_b['topic']}, entities={ctx_b['entities'][:5]}")

    # Ahora cada uno hace una query vaga
    r_a2 = k.ask("y el repercutido?", session_id="user_A")
    r_b2 = k.ask("y la seguridad social?", session_id="user_B")

    if r_a2:
        print(f"\n  User A: \"y el repercutido?\"")
        print(f"    → [{r_a2.level.value} {r_a2.score:.2f}] {r_a2.answer[:100].replace(chr(10),' ')}...")
    if r_b2:
        print(f"  User B: \"y la seguridad social?\"")
        print(f"    → [{r_b2.level.value} {r_b2.score:.2f}] {r_b2.answer[:100].replace(chr(10),' ')}...")

    # Verificar que no se mezclaron
    mezclado = False
    if ctx_a['topic'] == ctx_b['topic']:
        mezclado = True
    print(f"\n  ¿Se mezclaron? {'SÍ (ERROR)' if mezclado else 'NO — Perfecto'}")


def main():
    print("=" * 65)
    print("  KORE — TEST MEMORIA CONVERSACIONAL × CONTABILIZA-ES")
    print("=" * 65)

    # Crear motor
    k = Kore(domain="memoria_test", db_path=":memory:", config=FISCAL_CONFIG,
             max_turns=10, session_ttl=3600)

    # Cargar dataset
    print("\n── Carga del dataset ───────────────────────────────────────────")
    cargar_dataset(k)

    # Ejecutar conversaciones
    for conv in CONVERSACIONES:
        if conv["session_id"] and conv["turnos"]:
            ejecutar_conversacion(k, conv)

    # Test sesiones paralelas
    test_sesiones_paralelas(k)

    # Historial
    print(f"\n{'─' * 65}")
    print(f"  Historial de sesión 'asesor_maria'")
    print(f"{'─' * 65}")
    for turn in k.history("asesor_maria"):
        print(f"  Q: {turn['query'][:60]}")
        print(f"  A: {turn['answer'][:80]}...")
        print()

    # Stats
    print(f"\n{'=' * 65}")
    print(f"  ESTADÍSTICAS FINALES")
    print(f"{'=' * 65}")
    print(json.dumps(k.stats().to_dict(), indent=2, ensure_ascii=False))

    k.close()


if __name__ == "__main__":
    main()
