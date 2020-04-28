clustering:
	python cluster_persistence_diagrams.py /links/groups/borgwardt/Projects/Ephemeral/Partly\ Cloudy/tda/masked/persistence_diagrams/
	python cluster_persistence_diagrams.py -m top_persistence /links/groups/borgwardt/Projects/Ephemeral/Partly\ Cloudy/tda/masked/persistence_diagrams/

clustering_distances:
	# Infinity norm
	python cluster_persistence_diagrams_distance.py -m infinity_norm -p 1 /links/groups/borgwardt/Projects/Ephemeral/Partly\ Cloudy/tda/masked/persistence_diagrams
	python cluster_persistence_diagrams_distance.py -m infinity_norm -p 2 /links/groups/borgwardt/Projects/Ephemeral/Partly\ Cloudy/tda/masked/persistence_diagrams
	# Total persistence
	python cluster_persistence_diagrams_distance.py -p 1 /links/groups/borgwardt/Projects/Ephemeral/Partly\ Cloudy/tda/masked/persistence_diagrams
	python cluster_persistence_diagrams_distance.py -p 2 /links/groups/borgwardt/Projects/Ephemeral/Partly\ Cloudy/tda/masked/persistence_diagrams
	# Betti curves (no power parameter necessary)
	python cluster_persistence_diagrams_distance.py -m betti_curves /links/groups/borgwardt/Projects/Ephemeral/Partly\ Cloudy/tda/masked/persistence_diagrams

summary_statistics:
	python summary_statistics_persistence_diagrams.py -i /links/groups/borgwardt/Projects/Ephemeral/Partly\ Cloudy/tda/masked/persistence_diagrams/ -s total_persistence -s infinity_norm -p 1 -p 2 -o ../results/summary_statistics/masked.json

