import mlflow
import dagshub
import logging
from mlflow import MlflowClient

# --- 1. Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. Configuration ---
REPO_OWNER = 'gauravshuklaaaaa'
REPO_NAME = 'uber_demand_prediction'
REGISTERED_MODEL_NAME = 'uber_new_model'
ALIAS = "champion"
PROMOTION_STAGE = "Production"

def promote_champion_to_production():
    try:
        # --- 3. DagsHub & MLflow Init ---
        # Note: Ensure you have run 'dagshub login' in your terminal
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
        mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
        
        client = MlflowClient()
        logger.info(f"Connected to MLflow Tracking URI: {mlflow.get_tracking_uri()}")

        # --- 4. Get Version via Alias ---
        logger.info(f"Searching for model version with alias: '{ALIAS}'...")
        try:
            model_version_details = client.get_model_version_by_alias(
                name=REGISTERED_MODEL_NAME, 
                alias=ALIAS
            )
        except Exception as e:
            logger.error(f"Alias '{ALIAS}' nahi mila! Pehle DagsHub UI par jaakar alias assign karo.")
            return

        champion_v = model_version_details.version
        logger.info(f"Found Champion Alias on Version: {champion_v}")

        # --- 5. Transition Stage ---
        # Yahan aksar 403 aata hai agar permissions sahi na ho
        logger.info(f"Attempting to move Version {champion_v} to {PROMOTION_STAGE}...")
        
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=champion_v,
            stage=PROMOTION_STAGE,
            archive_existing_versions=True
        )
        
        logger.info("--- SUCCESS! ---")
        logger.info(f"Model version {champion_v} is now in '{PROMOTION_STAGE}' stage.")

    except mlflow.exceptions.MlflowException as me:
        if "403" in str(me):
            logger.error("Error 403: Write Access denied. Terminal mein 'dagshub login' karo ya DagsHub par settings check karo.")
        else:
            logger.error(f"MLflow Error: {me}")
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")

if __name__ == "__main__":
    promote_champion_to_production()