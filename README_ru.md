# nn-practice
Этот проект является практической проверкой моих знаний о нейронных сетях. Он исполнен с помощью NumPy. Примечательная особенность этой имплементации состоит в том, что эта модель поддерживает векторные фукнции активации, у которых выходные значения зависят от всех входных значений одновременно (например, Softmax).

## Требования
Данная модель имеет следующие зависимости:
* Python 3.6 или выше
* пакет NumPy

## Демонстрация
В этом репозитории есть ноутбук Jupyter с демонстрацией работы модели. Он запускает тренировку и классификацию на наборе изображений рукописных цифр MNIST.
Демонстрация также имеет дополнительные зависимости `pickle` и `gzip`.

## Ссылки
Пользовательский интерфейс модели, общая структура и стиль документации приблизительно адаптированы из https://github.com/mnielsen/neural-networks-and-deep-learning. Файлы `mnist.pkl.gz` и `mnist_loader_mod.py` для демонстрации взяты прямо из этого репозитория, притом последний модифицирован, чтобы работать с Python 3.
