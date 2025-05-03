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
      - train: Accuracy, Recall, F1-Score
      - test:  Accuracy, Recall, F1-Score
    """
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = iter(f)
        for line in lines:
            m_name = re.match(r'^\s*Ejecutando:\s*(.+)$', line)
            if not m_name:
                continue

            nombre = m_name.group(1).strip()
            time_sec = None
            metrics = []

            # Captura tiempo de entrenamiento
            for line2 in lines:
                m_time = re.match(r'^\s*Tiempo de entrenamiento[: ]+\s*([0-9]*\.?[0-9]+)', line2)
                if m_time:
                    time_sec = float(m_time.group(1))
                    break

            # Captura métricas (6 valores)
            for line2 in lines:
                if len(metrics) >= 6:
                    break
                if line2.strip().startswith('Loss'):
                    continue
                m_acc = re.match(r'^\s*Accuracy[: ]+\s*([0-9]*\.?[0-9]+)', line2)
                if m_acc:
                    metrics.append(float(m_acc.group(1)))
                    continue
                m_rec = re.match(r'^\s*Recall(?:.*)[: ]+\s*([0-9]*\.?[0-9]+)', line2)
                if m_rec:
                    metrics.append(float(m_rec.group(1)))
                    continue
                m_f1 = re.match(r'^\s*F1(?:[- ]Score.*)[: ]+\s*([0-9]*\.?[0-9]+)', line2)
                if m_f1:
                    metrics.append(float(m_f1.group(1)))
                    continue

            if len(metrics) == 6:
                train_acc, train_rec, train_f1, test_acc, test_rec, test_f1 = metrics
                entries.append({
                    'nombre':       nombre,
                    'acc test':     test_acc,
                    'recall test':  test_rec,
                    'f1 test':      test_f1,
                    'acc train':    train_acc,
                    'recall train': train_rec,
                    'f1 train':     train_f1,
                    'time (seg)':   time_sec
                })
            else:
                print(f" aviso: métricas incompletas para '{nombre}' (encontradas {len(metrics)})")

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Parsea un .out y exporta las métricas a un Excel con 4 decimales.")
    parser.add_argument('input', help="Ruta al fichero .out")
    parser.add_argument('-o', '--output',
                        help="Nombre del Excel de salida (por defecto: nombre del fichero de entrada con extensión .xlsx)")
    parser.add_argument('-a', '--average', type=int, default=0,
                        help="0=no summary, 1=media global, N=media cada N filas (por defecto 0)")
    args = parser.parse_args()

    # Determinar nombre de salida
    if args.output:
        output_file = args.output
    else:
        filename = os.path.basename(args.input)
        base, _ = os.path.splitext(filename)
        output_file = f"{base}.xlsx"

    datos = parse_out_file(args.input)
    if not datos:
        print("No se extrajeron métricas. Revisa el formato del .out.")
        return

    df = pd.DataFrame(datos, columns=[
        'nombre', 'acc test', 'recall test', 'f1 test',
        'acc train', 'recall train', 'f1 train', 'time (seg)'
    ])

    if args.average > 0:
        new_rows = []
        total = len(df)
        group = args.average

        if group == 1:
            # Media global al final, primero la media luego salto
            new_rows = df.to_dict('records')
            mean_vals = df.select_dtypes(include=['number']).mean()
            summary = mean_vals.to_dict()
            summary['nombre'] = 'media global'
            new_rows.append(summary)
            # fila en blanco
            blank = {col: ("" if col == 'nombre' else None) for col in df.columns}
            new_rows.append(blank)
        else:
            # Media cada N filas: primero bloque, luego media, luego salto
            for i in range(0, total, group):
                chunk = df.iloc[i:i+group]
                new_rows.extend(chunk.to_dict('records'))
                if len(chunk) == group:
                    # fila resumen
                    mean_vals = chunk.select_dtypes(include=['number']).mean()
                    summary = mean_vals.to_dict()
                    summary['nombre'] = f"media filas {i+1}-{i+group}"
                    new_rows.append(summary)
                    # fila en blanco
                    blank = {col: ("" if col == 'nombre' else None) for col in df.columns}
                    new_rows.append(blank)
        df = pd.DataFrame(new_rows, columns=df.columns)

    df.to_excel(output_file, index=False, float_format="%.4f")
    print(f"Excel generado: {output_file}")

if __name__ == '__main__':
    main()
