data
- /readme_img: result of setup_test.py. For the README
- /templates: hand-card templates used to identifiy which cards are in hand
- /test: images used to test the model. Can be passed into test_infer_on_image.py

models
- left_70.zip: last training checkpoint for the model on the left of my screen
- right 70.zip: last training checkpoint for the model on the right of my screen
- vision.pt: weights for the YOLO v8 vision model

notebooks
- Ran in colab to fine-tune YOLO v8 on my 2 datasets

src
- collect_data.py: my self-play training script
- env.py: Gymnasium environments (ClashEnv and ReplayEnv)
- eval_checkpoint.py: self-play, but for evaluation. Loads left_70.zip and play against untrained (random) model
- getState.py: all helpers involving perception. Also encodes all information into state vectors.
- infer.py: runs one instance of the bot for evaluation against a human opponent.
- setup_test.py: script that confirms the ROI definitions if trying to run on a new laptop.
- takeAction.py: all helper functions involving actions (clicking, action-mask, action-decoding)
- **test_infer_on_images: file for the grades to run the model without having to deal with setup**
- train.py: offline training for the PPO agent using buffers gathered from collect_data.py
- vars.py: all global variable definitions (pixel positions, card costs, card names)
