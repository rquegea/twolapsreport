#!/usr/bin/env python3
import os
import re
import sys
from typing import Iterator

COPY_START_RE = re.compile(r"^COPY\s+([^\s]+)\s*\((.*?)\)\s+FROM\s+stdin;\s*$", re.IGNORECASE)


def generate_merge_sql(lines: Iterator[str]) -> Iterator[str]:
    in_copy = False
    table_name = ""
    column_list = ""
    temp_name = ""

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if not in_copy:
            m = COPY_START_RE.match(line)
            if m:
                in_copy = True
                table_name = m.group(1)
                column_list = m.group(2)
                temp_name = f"import_tmp_{table_name.replace('.', '_')}"
                # Crear tabla temporal vacía clonando estructura
                yield f"CREATE TEMP TABLE {temp_name} AS TABLE {table_name} WITH NO DATA;"
                # Redirigir COPY a la temporal
                yield f"COPY {temp_name} ({column_list}) FROM stdin;"
            else:
                # Pasar líneas fuera de COPY tal cual
                yield line
        else:
            # Estamos dentro de COPY
            yield line
            if line == r"\.":
                # Cerrar el bloque: mover a tabla real ignorando conflictos y borrar temp
                yield (
                    f"INSERT INTO {table_name} ({column_list}) "
                    f"SELECT {column_list} FROM {temp_name} ON CONFLICT DO NOTHING;"
                )
                yield f"DROP TABLE {temp_name};"
                in_copy = False
                table_name = ""
                column_list = ""
                temp_name = ""

    # Si el dump estaba mal y terminó dentro de COPY, lo dejamos tal cual


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Uso: transform_dump_for_merge.py <input.sql> <output.sql>",
            file=sys.stderr,
        )
        return 2

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isfile(input_path):
        print(f"No existe: {input_path}", file=sys.stderr)
        return 1

    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for out_line in generate_merge_sql(fin):
            fout.write(out_line + "\n")

    print(f"✅ Generado: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
