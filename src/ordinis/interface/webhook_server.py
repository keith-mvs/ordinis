import asyncio
import logging
import os
import json
from aiohttp import web
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ordinis.interface.webhook")

# Configuration
FINNHUB_SECRET = os.getenv("FINNHUB_WEBHOOK_SECRET", "d55jsapr01qu4cch9iv0")
HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
PORT = int(os.getenv("WEBHOOK_PORT", "8080"))

async def handle_finnhub_webhook(request: web.Request) -> web.Response:
    """
    Handle incoming Finnhub webhook events.
    
    Requirement: "To acknowledge receipt of an event, your endpoint must return 
    a 2xx HTTP status code. Acknowledge events prior to any logic that needs 
    to take place to prevent timeouts."
    """
    # 1. Verify Authentication Header
    secret = request.headers.get("X-Finnhub-Secret")
    if secret != FINNHUB_SECRET:
        logger.warning(f"Unauthorized webhook attempt. Secret: {secret}")
        return web.Response(status=401, text="Unauthorized")

    # 2. Read payload (non-blocking if possible, but we need the data)
    try:
        # We read the content but don't process it yet to ensure fast response
        if request.body_exists:
            content = await request.read()
            payload = content.decode('utf-8')
            
            # Put into queue for background processing
            await request.app['event_queue'].put({
                "source": "finnhub",
                "payload": payload,
                "headers": dict(request.headers)
            })
            
            logger.info("Received Finnhub event, queued for processing")
        else:
            logger.warning("Received empty Finnhub event")

    except Exception as e:
        logger.error(f"Error reading webhook body: {e}")
        # Even if reading fails, we might want to return 400, but to prevent 
        # disabling the endpoint, specific errors might still warrant a 200 
        # if it's a malformed payload but valid auth. 
        # For now, let's return 400 for bad requests.
        return web.Response(status=400, text="Bad Request")

    # 3. Acknowledge Receipt (2xx)
    return web.Response(status=200, text="Event Received")

async def event_processor(app):
    """Background task to process events from the queue."""
    logger.info("Starting event processor...")
    queue = app['event_queue']
    
    try:
        while True:
            event = await queue.get()
            try:
                # Placeholder for actual processing logic
                # In a real system, this would publish to the StreamingBus
                logger.info(f"Processing event: {event.get('source')}")
                
                payload = event.get('payload')
                try:
                    data = json.loads(payload)
                    logger.info(f"Event Data: {json.dumps(data, indent=2)}")
                except json.JSONDecodeError:
                    logger.info(f"Raw Payload: {payload}")
                    
            except Exception as e:
                logger.error(f"Error processing event: {e}")
            finally:
                queue.task_done()
    except asyncio.CancelledError:
        logger.info("Event processor cancelled")

async def start_background_tasks(app):
    app['event_queue'] = asyncio.Queue()
    app['processor'] = asyncio.create_task(event_processor(app))

async def cleanup_background_tasks(app):
    app['processor'].cancel()
    await app['processor']

def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post('/webhook/finnhub', handle_finnhub_webhook)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    return app

def main():
    app = create_app()
    logger.info(f"Starting Webhook Server on {HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT)

if __name__ == '__main__':
    main()
