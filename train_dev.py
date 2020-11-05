from learner import QTLearner

if __name__ == '__main__':
    learner = QTLearner.create_from_conf('src/config_dev.py')
    learner.fit()
