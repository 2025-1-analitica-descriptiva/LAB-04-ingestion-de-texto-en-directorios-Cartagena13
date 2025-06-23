# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""
import pandas as pd
from pathlib import Path

def pregunta_01():
    ROOT        = Path('files/input/input')          # carpeta raíz descomprimida
    TRAIN_DIR   = ROOT / 'train'                     # …/train/…
    TEST_DIR    = ROOT / 'test'                      # …/test/…
    OUTPUT_DIR  = Path('files/output')                     # carpeta donde guardaremos .csv
    OUTPUT_DIR.mkdir(exist_ok=True)                  # la crea si no existe
    def build_split_dataframe(split_dir: Path) -> pd.DataFrame:
        """
        Recorre un directorio (train o test) y arma un DataFrame con columnas:
        phrase (str)  ·  sentiment (str)
        """
        registros = []

        # rglob('*/*.txt') → todos los .txt que estén exactamente UN nivel más abajo
        for txt_path in split_dir.rglob('*.txt'):
            # txt_path = …/train/positive/0003.txt   →   txt_path.parent.name = 'positive'
            target = txt_path.parent.name.lower()      # garantiza minúsculas
            phrase    = txt_path.read_text(encoding='utf-8').strip()

            registros.append({'phrase': phrase, 'target': target})

        return pd.DataFrame(registros, columns=['phrase', 'target'])
    # ----------- Generar DataFrames ----------------------------------------------
    train_df = build_split_dataframe(TRAIN_DIR)
    test_df  = build_split_dataframe(TEST_DIR)

    # ----------- Guardar en CSV ---------------------------------------------------
    train_df.to_csv(OUTPUT_DIR / 'train_dataset.csv', index=False)
    test_df.to_csv(OUTPUT_DIR / 'test_dataset.csv',  index=False)
    return pregunta_01

    """
    La información requerida para este laboratio esta almacenada en el
    archivo "files/input.zip" ubicado en la carpeta raíz.
    Descomprima este archivo.

    Como resultado se creara la carpeta "input" en la raiz del
    repositorio, la cual contiene la siguiente estructura de archivos:


    ```
    train/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    test/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    ```

    A partir de esta informacion escriba el código que permita generar
    dos archivos llamados "train_dataset.csv" y "test_dataset.csv". Estos
    archivos deben estar ubicados en la carpeta "output" ubicada en la raiz
    del repositorio.

    Estos archivos deben tener la siguiente estructura:

    * phrase: Texto de la frase. hay una frase por cada archivo de texto.
    * sentiment: Sentimiento de la frase. Puede ser "positive", "negative"
      o "neutral". Este corresponde al nombre del directorio donde se
      encuentra ubicado el archivo.

    Cada archivo tendria una estructura similar a la siguiente:

    ```
    |    | phrase                                                                                                                                                                 | target   |
    |---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
    |  0 | Cardona slowed her vehicle , turned around and returned to the intersection , where she called 911                                                                     | neutral  |
    |  1 | Market data and analytics are derived from primary and secondary research                                                                                              | neutral  |
    |  2 | Exel is headquartered in Mantyharju in Finland                                                                                                                         | neutral  |
    |  3 | Both operating profit and net sales for the three-month period increased , respectively from EUR16 .0 m and EUR139m , as compared to the corresponding quarter in 2006 | positive |
    |  4 | Tampere Science Parks is a Finnish company that owns , leases and builds office properties and it specialises in facilities for technology-oriented businesses         | neutral  |
    ```


    """
