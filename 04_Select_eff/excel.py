#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
import os
import pandas as pd

def parse_out_file(path):
    """
    Recorre el fichero línea a línea, y cada vez que encuentra
    'Ejecutando: <nombre>' arranca la captura de:
      - tiempo de entrenamiento (segundos)
      - métricas valid: Accuracy, Recall, F1-Score
      - métricas test:  Accuracy, Recall, F1-Score
    """
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        m_name = re.match(r'^\s*Ejecutando:\s*(.+)$', line)
        if not m_name:
            i += 1
            continue

        nombre = m_name.group(1).strip()
        time_sec = None
        # inicializamos métricas a None
        train_acc = train_rec = train_f1 = None
        test_acc  = test_rec  = test_f1  = None

        # 1) buscar tiempo de entrenamiento
        j = i + 1
        while j < len(lines):
            m_time = re.search(r'Tiempo de entrenamiento[: ]+([0-9]*\.?[0-9]+)', lines[j])
            if m_time:
                time_sec = float(m_time.group(1))
                break
            j += 1

        # 2) buscar métricas valid y test
        k = j + 1
        while k < len(lines):
            l = lines[k]
            # valid / entrenamiento
            m = re.search(r'Accuracy\s*\(valid\)[: ]+([0-9]*\.?[0-9]+)', l)
            if m:
                train_acc = float(m.group(1))
            m = re.search(r'Recall\s*\(valid\)[: ]+([0-9]*\.?[0-9]+)', l)
            if m:
                train_rec = float(m.group(1))
            m = re.search(r'F1[- ]Score\s*\(valid\)[: ]+([0-9]*\.?[0-9]+)', l)
            if m:
                train_f1 = float(m.group(1))

            # test
            m = re.search(r'Accuracy\s*\(test\)[: ]+([0-9]*\.?[0-9]+)', l)
            if m:
                test_acc = float(m.group(1))
            m = re.search(r'Recall\s*\(test\)[: ]+([0-9]*\.?[0-9]+)', l)
            if m:
                test_rec = float(m.group(1))
            m = re.search(r'F1[- ]Score\s*\(test\)[: ]+([0-9]*\.?[0-9]+)', l)
            if m:
                test_f1 = float(m.group(1))

            # si ya tenemos todo, salimos
            if None not in (train_acc, train_rec, train_f1, test_acc, test_rec, test_f1):
                break

            k += 1

        # verificar resultados
        if None in (train_acc, train_rec, train_f1, test_acc, test_rec, test_f1):
            print(f" aviso: métricas incompletas para '{nombre}'")
        entries.append({
            'nombre':       nombre,
            'acc train':    train_acc,
            'recall train': train_rec,
            'f1 train':     train_f1,
            'acc test':     test_acc,
            'recall test':  test_rec,
            'f1 test':      test_f1,
            'time (seg)':   time_sec
        })

        # avanzamos el índice para seguir buscando nuevos bloques
        i = k + 1

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Parsea un .out y exporta las métricas a un Excel con 4 decimales.")
    parser.add_argument('input', help="Ruta al fichero .out")
    parser.add_argument('-o', '--output',
                        help="Nombre del Excel de salida (por defecto: entrada.xlsx)")
    parser.add_argument('-a', '--average', type=int, default=0,
                        help="0=no summary, 1=media global, N=media cada N filas (por defecto 0)")
    args = parser.parse_args()

    # determinar nombre de salida
    if args.output:
        output_file = args.output
    else:
        base, _ = os.path.splitext(os.path.basename(args.input))
        output_file = f"{base}.xlsx"

    datos = parse_out_file(args.input)
    if not datos:
        print("No se extrajeron métricas. Revisa el formato del .out.")
        return

    df = pd.DataFrame(datos, columns=[
        'nombre',
        'acc train', 'recall train', 'f1 train',
        'acc test',  'recall test',  'f1 test',
        'time (seg)'
    ])

    # resumen por bloques
    if args.average > 0:
        rows = []
        n = args.average
        total = len(df)

        if n == 1:
            rows = df.to_dict('records')
            m = df.select_dtypes('number').mean()
            summary = m.to_dict()
            summary['nombre'] = 'media global'
            rows.append(summary)
            rows.append({col: ("" if col=='nombre' else None) for col in df.columns})
        else:
            for start in range(0, total, n):
                chunk = df.iloc[start:start+n]
                rows.extend(chunk.to_dict('records'))
                if len(chunk) == n:
                    m = chunk.select_dtypes('number').mean().to_dict()
                    m['nombre'] = f"media filas {start+1}-{start+n}"
                    rows.append(m)
                    rows.append({col: ("" if col=='nombre' else None) for col in df.columns})
        df = pd.DataFrame(rows, columns=df.columns)

    df.to_excel(output_file, index=False, float_format="%.4f")
    print(f"Excel generado: {output_file}")


if __name__ == '__main__':
    main()
