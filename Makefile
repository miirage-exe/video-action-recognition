# Generate train.py from the notebook, stripping Colab-only cells (tagged docker-skip).
# Run this whenever the notebook changes, before rebuilding the Docker image.

train.py: ActionRecognition_GPU.ipynb
	jupyter nbconvert --to script ActionRecognition_GPU.ipynb --stdout \
                --TagRemovePreprocessor.enabled=True \
                --TagRemovePreprocessor.remove_cell_tags='["docker-skip"]' > train.py

.PHONY: clean
clean:
	rm -f train.py
