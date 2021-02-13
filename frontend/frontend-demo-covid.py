import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
# Esto es un comentario
import numpy
import os
import pandas

from sklearn.model_selection import train_test_split

from environment import DATA_DIR


formatting = lambda x: '{:05d}'.format(x)

def generate_splits(data):
    """
        Crear splits train/validation/test
    Args:
        data: (N,3) shape pandas.DataFrame
    Return:
        None
    """
    data["name"] = ["dummy_name" + str(i+1) for i in range(len(data))]
    # Generamos 70% train, 30% test a partir de la tabla data
    train, test = train_test_split(data, test_size=0.3)
    # Dentro de train, generamos train y validation
    # Del 70% de train, un 85% siguen en train
    # y un 15% van a validacion
    train, val = train_test_split(train, test_size=0.15)
    # Guardar datos train/val/test en un fichero externo
    # DATA_DIR = 'datasets/'
    train.to_csv(os.path.join(
        DATA_DIR, 'COVID-DEMO', 'training.txt'),
        index=False, header=None, sep='\t'
    )
    val.to_csv(os.path.join(
        DATA_DIR, 'COVID-DEMO', 'validation.txt'),
        index=False, header=None, sep='\t'
    )
    test.to_csv(os.path.join(
        DATA_DIR, 'COVID-DEMO', 'test.txt'),
        index=False, header=None, sep='\t'
    )

def _main_():
    path_to_file = sys.argv[1]
    data = pandas.read_csv(path_to_file, sep='\t', header=None)
    data = data.rename(columns={
        0: "number",
        1: "label",
        2: "text"
    })
    data["number"] = data["number"].apply(formatting)

    generate_splits(data)

if __name__ == "__main__":
    _main_()