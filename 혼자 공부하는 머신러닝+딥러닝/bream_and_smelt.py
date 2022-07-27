'''
Chapter 01. 도미와 빙어
'''
import sys
import time
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class Bream: # 도미
    '''
    도미에 대한 내용들이다.
    '''
    # 길이(cm)
    __BREAM_LENGTH = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7,
                      31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5,
                      34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0,
                      38.5, 38.5, 39.5, 41.0, 41.0]

    # 무게(g)
    __BREAM_WEIGHT = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0,
                      475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0,
                      575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0,
                      920.0, 955.0, 925.0, 975.0, 950.0]

    def __init__(self) -> None:
        return

    def draw_scatter(self, png_name: str) -> None:
        '''
        도미의 산점도를 그리고 png 로 저장한다.
        '''
        plt.figure()  # 새로운 Figure 객체를 생성
        plt.scatter(self.__BREAM_LENGTH, self.__BREAM_WEIGHT)  # 산점도를 그린다.
        plt.xlabel('length')  # 산점도의 x축 라벨의 이름을 지정한다.
        plt.ylabel('weigth')  # 산점도의 y축 라벨의 이름을 지정한다.
        plt.savefig(png_name + '.png')  # plt.show() 대신 png 로 저장한다.

    def get_data(self) -> list:
        '''
        도미의 데이터를 리턴한다.
        '''
        return [self.__BREAM_LENGTH, self.__BREAM_WEIGHT]

class Smelt: # 빙어
    '''
    빙어에 대한 내용들이다.
    '''
    # 길이(cm)
    __SMELT_LENGTH = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2,
                      12.4, 13.0, 14.3, 15.0]

    # 무게(g)
    __SMELT_WEIGHT = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2,
                      13.4, 12.2, 19.7, 19.9]

    def __init__(self) -> None:
        return

    def draw_scatter(self, png_name: str) -> None:
        '''
        빙어의 산점도를 그리고 png 로 저장한다.
        '''
        plt.figure()  # 새로운 Figure 객체를 생성
        plt.scatter(self.__SMELT_LENGTH, self.__SMELT_WEIGHT)  # 산점도를 그린다.
        plt.xlabel('length')  # 산점도의 x축 라벨의 이름을 지정한다.
        plt.ylabel('weigth')  # 산점도의 y축 라벨의 이름을 지정한다.
        plt.savefig(png_name + '.png')  # plt.show() 대신 png 로 저장한다.

    def get_data(self) -> list:
        '''
        빙어의 데이터를 리턴한다.
        '''
        return [self.__SMELT_LENGTH, self.__SMELT_WEIGHT]

class BreamAndSmelt: # 도미와 빙어
    '''
    도미와 빙어에 대한 내용들이다.
    '''
    class Target(Enum):
        '''
        생선종류에 대한 Enum 이다.
        '''
        BREAM = 0
        SMELT = 1

    __LENGTH_IDX = 0
    __WEIGHT_IDX = 1

    def __init__(self, bream_data: list, smelt_data: list) -> None:
        self.__bream_data = bream_data
        self.__smelt_data = smelt_data
        self.__k_neighbors_classifier = KNeighborsClassifier()

    def draw_scatter(self, png_name: str) -> None:
        '''
        도미와 빙어의 산점도를 함께 그리고 png 로 저장한다.
        '''
        plt.figure()  # 새로운 Figure 객체를 생성

        # 도미의 산점도를 그린다.
        plt.scatter(self.__bream_data[self.__LENGTH_IDX],
                    self.__bream_data[self.__WEIGHT_IDX])

        # 빙어의 산점도를 그린다.
        plt.scatter(self.__smelt_data[self.__LENGTH_IDX],
                    self.__smelt_data[self.__WEIGHT_IDX])

        plt.xlabel('length')  # 산점도의 x축 라벨의 이름을 지정한다.
        plt.ylabel('weigth')  # 산점도의 y축 라벨의 이름을 지정한다.
        plt.savefig(png_name + '.png')  # plt.show() 대신 png 로 저장한다.

    def learning_data(self, target: Target) -> None:
        '''
        k-최근접이웃(k-neighbors classifier) 알고리즘으로 데이터를 학습한다.
        '''
        # 도미와 빙어의 데이터를 합친다.
        bream_and_smelt_length = self.__bream_data[self.__LENGTH_IDX] + \
                                 self.__smelt_data[self.__LENGTH_IDX]
        bream_and_smelt_weigth = self.__bream_data[self.__WEIGHT_IDX] + \
                                 self.__smelt_data[self.__WEIGHT_IDX]

        # 합친 데이터를 학습이 가능한 데이터셋으로 변환을 한다.
        learning_data = [
            [length, weigth] for length, weigth in zip(bream_and_smelt_length, bream_and_smelt_weigth)
        ]

        # 학습 데이터셋에 대한 정답 데이터셋을 만든다.
        if target is self.Target.BREAM:
            answer_data = [1]*35 + [0]*14  # 도미를 찾을 거니까 도미가 1, 빙어가 0
        else:
            answer_data = [0]*35 + [1]*14  # 빙어를 찾을 거니까 도미가 0, 빙어가 1

        # k-최근접이웃 알고리즘으로 학습하여 모델을 만든다.
        start = time.process_time()
        self.__k_neighbors_classifier.fit(learning_data, answer_data)
        end = time.process_time()
        print('learning time:', end-start, 'second')

        # for n in range(5, 50):
        #     self.set_n_neighbors(n)
        #     score = self.check_data(learning_data, answer_data)
        #     if score < 1:
        #         print('score is ' + str(score) + '. n_neighbors: ' + str(n))

    def check_data(self, data: list, answer: list) -> float:
        '''
        학습된 모델로 data 를 평가한다.
        '''
        return float(self.__k_neighbors_classifier.score(data, answer))

    def predict_data(self, data: list) -> list:
        '''
        학습된 모델로 data 를 예측한다.
        '''
        return list(self.__k_neighbors_classifier.predict(data))

    def set_n_neighbors(self, num: int) -> None:
        '''
        n_neighbors 값을 설정한다.
        '''
        self.__k_neighbors_classifier.n_neighbors = num

def main() -> None:
    '''
    main 함수
    '''
    bream = Bream()
    bream.draw_scatter('bream_scatter')

    smelt = Smelt()
    smelt.draw_scatter('smelt_scatter')

    bream_and_smelt = BreamAndSmelt(bream.get_data(), smelt.get_data())
    bream_and_smelt.draw_scatter('bream_and_smelt_scatter')

    bream_and_smelt.learning_data(BreamAndSmelt.Target.BREAM)

    data: list = [[25.4, 240.0], [25.4, 20.0]]
    answer: list = [1, 1]

    print('data is', data)
    print('answer is', answer)

    score = bream_and_smelt.predict_data(data)
    print('predict is', score)

    score = bream_and_smelt.check_data(data, answer)
    print('score is', score)

if __name__ == '__main__':
    py_ver_major: str = str(sys.version_info[0])
    py_ver_minor: str = str(sys.version_info[1])
    py_ver_micro: str = str(sys.version_info[2])
    print('python_version: ' + '.'.join([py_ver_major, py_ver_minor, py_ver_micro]) + '\n')
    main()
