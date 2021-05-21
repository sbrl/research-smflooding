import io
from ..io.settings import settings_get


class CategoryCalculator:
	"""Calculates the category of a given string of text based on the number of occurences of category items."""
	
	def __init__(self):
		"""Creates a new CategoryCalculator instance."""
		self.settings = settings_get()
		
		self.count = None
		self.load()
	
	
	def load(self):
		"""Loads the category data into memory."""
		handle = io.open(self.settings.data.paths.categories, "r")
		
		self.categories = []
		self.markers = ""
		
		i = -1
		for line in handle:
			i += 1
			parts = line.strip().split("\t")
			
			self.categories.append({
				"i": i,
				"name": parts[0],
				"glyphs": list(parts[1])
			})
			self.markers += parts[1]
		
		
		self.count = i + 1
	
	def get_category_index(self, text: str):
		"""
		Calculates the index of the category a given string of text belongs to.
		Returns None if no category matches.
		"""
		weights = {}
		
		for category in self.categories:
			for glyph in category["glyphs"]:
				if not category["i"] in weights:
					weights[category["i"]] = 0
				
				weights[category["i"]] += text.count(glyph)
		
		max_i = None
		max_value = -1
		for i in weights:
			if weights[i] > max_value:
				max_i = i
				max_value = weights[i]
		
		return max_i
	
	def strip_markers(self, text: str):
		"""Removes all the markers from the given stringt that we use to decide on which category a string goes in."""
		return text.translate(text.maketrans("", "", self.markers))
	
	def get_category_name(self, text: str):
		"""Calculates the name of the category that the given string of text belongs to."""
		return self.index2name(self.get_category_index(text))
		
	def index2name(self, i: int):
		"""Returns the name for a given category index."""
		for cat in self.categories:
			if i == cat["i"]:
				return cat["name"]
	
	def name2index(self, name: str):
		"""Returns the index for  given category name."""
		for cat in self.categories:
			if name == cat["name"]:
				return cat["i"]
