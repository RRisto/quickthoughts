from learner import QTLearner
from src.config import CONFIG

learner = QTLearner(CONFIG['checkpoint_dir'], CONFIG['embedding'], CONFIG['data_path'], CONFIG['batch_size'],
                    CONFIG['hidden_size'], CONFIG['lr'], CONFIG['resume'], CONFIG['num_epochs'],
                    CONFIG['norm_threshold'])

learner.fit()
