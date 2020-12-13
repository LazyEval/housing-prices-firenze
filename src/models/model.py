import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


class Model:
	def __init__(self, model=None):
		"""Set model and name attributes."""
		self.model = model

		if type(self.model).__name__ == 'Pipeline':
			self.name = type(self.model.named_steps['model']).__name__
		else:
			self.name = type(self.model).__name__

	def get_params(self):
		"""Return model parameters."""
		return self.model.get_params()

	def train(self, X, y):
		"""Train model."""
		self.model.fit(X, y)

	@classmethod
	def tune(cls, model, X, y, param_grid, cv=5):
		"""Instantiate class with tuned model."""
		grid_search = GridSearchCV(model, cv=cv, param_grid=param_grid)
		grid_search.fit(X, y)
		return cls(grid_search.best_estimator_)

	def cv_score(self, X, y, scoring, cv=5):
		"""Return model cross-validated score."""
		return cross_val_score(self.model, X, y, cv=cv, scoring=scoring).mean()

	def predict(self, X):
		"""Return predictions from model."""
		y_pred = self.model.predict(X)
		return y_pred

	def save(self, path):
		"""Save model as pickle file."""
		with open(path, 'wb') as f:
			joblib.dump(self.model, f)
			joblib.dump(self.name, f)

	@staticmethod
	def load(path):
		"""Load model from pickle file and return it."""
		model = Model()
		with open(path, 'rb') as f:
			model.model = joblib.load(f)
			model.name = joblib.load(f)
		return model
