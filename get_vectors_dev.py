from learner import QTLearner

if __name__ == '__main__':
    learner = QTLearner.create_from_checkpoint('checkpoints/dev')
    learner.predict(['hi my name is sanyam .'])
