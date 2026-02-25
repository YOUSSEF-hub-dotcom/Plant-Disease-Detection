import logging

def setup_logging():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("plant_disease_pipeline.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )