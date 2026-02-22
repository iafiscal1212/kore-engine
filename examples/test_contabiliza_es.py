#!/usr/bin/env python3
"""
Kore × CONTABILIZA-ES — Test con dataset real
══════════════════════════════════════════════
Descarga el dataset de HuggingFace y lo ingesta en Kore.
Luego lanza batería de queries fiscales + no-fiscales.

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
FILES = ["dataset_completo.jsonl", "train.jsonl"]

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
    learn_from_feedback=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA
# ─────────────────────────────────────────────────────────────────────────────

def descargar_dataset() -> Path:
    """Descarga el dataset y lo guarda en un temporal unificado."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    all_lines = []
    seen = set()

    for filename in FILES:
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

    # Guardar en temporal
    tmp = Path(tempfile.mktemp(suffix='.jsonl'))
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines))
    print(f"  Total: {len(all_lines)} lineas unicas → {tmp}")
    return tmp


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  KORE × CONTABILIZA-ES — Test con dataset real")
    print("=" * 70)
    print()

    # 1. Descargar
    print("── Descarga ────────────────────────────────────────────────────")
    dataset_path = descargar_dataset()
    print()

    # 2. Crear Kore e ingestar
    print("── Ingesta ─────────────────────────────────────────────────────")
    t0 = time.perf_counter()

    k = Kore(domain="contabiliza_es", db_path=":memory:", config=FISCAL_CONFIG)
    result = k.ingest(dataset_path)

    elapsed_ingest = time.perf_counter() - t0
    print(f"  Resultado: {result['loaded']} cargadas, {result['errors']} errores")
    print(f"  Tiempo: {elapsed_ingest:.1f}s")
    print()

    # 3. Queries fiscales (deberían dar hit)
    print("── Queries fiscales ────────────────────────────────────────────")
    fiscal_queries = [
        "Genera asiento de compra con IVA",
        "asiento de nómina con seguridad social",
        "cómo se hace el cierre del ejercicio",
        "qué es la regularización de existencias",
        "tipos de IVA en España",
        "factura rectificativa por devolución",
        "IVA intracomunitario",
        "retención de IRPF en nómina",
        "cuentas anuales obligatorias",
        "depósito cuentas registro mercantil",
        "qué dice el BOICAC sobre amortización",
        "alta en seguridad social",
        "cómo se contabiliza un leasing",
        "amortización de inmovilizado material",
        "prorrata del IVA",
    ]

    hits = 0
    t_queries = time.perf_counter()

    for q in fiscal_queries:
        r = k.ask(q)
        if r:
            hits += 1
            preview = r.answer[:90].replace("\n", " ")
            print(f"  HIT  [{r.level.value:8s} {r.score:.2f}]  {q}")
            print(f"       → {preview}...")
        else:
            print(f"  MISS                  {q}")
        print()

    elapsed_fiscal = (time.perf_counter() - t_queries) * 1000

    # 4. Queries NO fiscales (deberían dar miss)
    print("── Queries NO fiscales (deben ser MISS) ─────────────────────────")
    non_fiscal = [
        "receta de tortilla de patatas",
        "cuánto cuesta un vuelo a Londres",
        "quién ganó la liga 2025",
        "cómo se entrena un perro",
        "mejor restaurante de Barcelona",
    ]

    false_positives = 0
    for q in non_fiscal:
        r = k.ask(q)
        if r:
            false_positives += 1
            print(f"  FALSE HIT [{r.level.value:8s} {r.score:.2f}]  {q}")
        else:
            print(f"  MISS OK                              {q}")
    print()

    # 5. Resumen
    total_queries = len(fiscal_queries) + len(non_fiscal)
    elapsed_total = (time.perf_counter() - t_queries) * 1000

    print("=" * 70)
    print(f"  RESULTADOS")
    print("=" * 70)
    print(f"  Dataset:           {result['loaded']} entries")
    print(f"  Ingesta:           {elapsed_ingest:.1f}s")
    print(f"  Queries fiscales:  {hits}/{len(fiscal_queries)} hits")
    print(f"  False positives:   {false_positives}/{len(non_fiscal)}")
    print(f"  Tiempo queries:    {elapsed_total:.1f}ms ({elapsed_total/total_queries:.1f}ms/query)")
    print(f"  Hit rate fiscal:   {hits/len(fiscal_queries):.0%}")
    print("=" * 70)
    print()

    # 6. Stats
    print("Stats Kore:")
    print(json.dumps(k.stats().to_dict(), indent=2, ensure_ascii=False))

    # Cleanup
    k.close()
    dataset_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
