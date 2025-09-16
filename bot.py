# sweep_bot.py
import json
import os
import logging
from datetime import datetime, timedelta, timezone

from typing import Dict, List

import discord
from discord.ext import tasks

import openai

client = openai.Client()
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("sweep-bot")

# Router - Evaluates a sweep of messages and decides what to do with them, by instructing an LLM to evaluate. The LLM is expected to return a JSON object with instructions.
# Router can do a number of things, including absolutely nothing "noop" if nothing of interest is found. It can send a message to a channel. It can also save a memo. It can do up to max actions per sweep. The bot will be notified of how many actions have been spent and how many including CURRENT run remain. Custom command execution with command and payloud is also supported. They should of course be listed when available but right now that list is empty. Noop if called will exit the loop
router_decision_schema = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["noop", "send_message", "save_memo", "custom_command"]
                    },
                    "channel_id": {"type": "integer"},
                    "message": {"type": "string"},
                    "memo": {"type": "string"},
                    "command": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["type"],
                "additionalProperties": False
            }
        }
    },
    "required": ["actions"],
    "additionalProperties": False
}

def save_memo(memo: str):
    # Save to bot_data/memos.jsonl
    memo_json = {"timestamp": datetime.now(timezone.utc).isoformat(), "memo": memo}
    os.makedirs("bot_data", exist_ok=True)
    with open("bot_data/memos.jsonl", "a") as f:
        f.write(json.dumps(memo_json) + "\n")
    log.info(f"Memo saved: {memo}")
    # Implement your memo saving logic here
    
async def execute_custom_command(command: str, payload: dict):
    # Placeholder for custom command execution logic
    log.info(f"Executing custom command: {command} with payload: {payload}")
    # Implement your custom command logic here
            

# Storage class for a sweep.
class Sweep:
    def __init__(self, channel_id: int, after: 'datetime.datetime'):
        self.channel_id = channel_id
        self.after = after
        self.messages: List[discord.Message] = []

    def add_message(self, message: discord.Message):
        self.messages.append(message)

    # Converts the messages, including any relevant metadata, into a format suitable for LLM processing.
    def to_llm_readable(self):
        return [
            {
                "role": "user" if msg.author.bot is False else "system",
                "content": msg.content,
                "author": {
                    "id": msg.author.id,
                    "name": msg.author.name,
                    "discriminator": msg.author.discriminator,
                    "bot": msg.author.bot,
                },
                "timestamp": msg.created_at.isoformat(),
                "attachments": [att.url for att in msg.attachments],
            }
            for msg in self.messages
            if msg.content or msg.attachments
        ]


class SweepBot(discord.Client):
    def __init__(
        self,
        *,
        bot_name: str = "Helpful Discord Bot",
        monitored_channel_ids: List[int],
        sweep_interval_seconds: int = 300,          # how often to sweep
        startup_lookback_minutes: int = 30,         # how far back to look on startup
        max_messages_per_channel: int = 500,        # cap per sweep to be gentle on rate limits
        router_model = "gpt-5-mini",
        max_iterations = 2,
        max_actions_per_sweep = 3,
        **options
    ):
        super().__init__(**options)
        self.bot_name = bot_name
        self.monitored_channel_ids = monitored_channel_ids
        self.startup_lookback = timedelta(minutes=startup_lookback_minutes)
        self.max_messages_per_channel = max_messages_per_channel
        self.router_model = router_model
        self.max_iterations = max_iterations
        self.max_actions_per_sweep = max_actions_per_sweep
        self._last_seen: Dict[int, datetime] = {}

        # set initial interval; can be changed at runtime if desired
        self.sweep_recent_activity.change_interval(seconds=sweep_interval_seconds)

    async def setup_hook(self):
        # start the loop once the client is ready
        self.sweep_recent_activity.start()

    async def on_ready(self):
        log.info(f"Logged in as {self.user} (ID: {self.user.id})")
        log.info(f"Monitoring {len(self.monitored_channel_ids)} channel(s).")

    async def send_message(self, channel_id: int, content: str):
        """Send a message to the specified Discord channel."""
        # Get the Discord channel by ID
        channel = discord.utils.get(self.guilds[0].channels, id=channel_id)
        if channel is None:
            try:
                channel = await client.fetch_channel(channel_id)
            except Exception as e:
                log.error(f"Failed to fetch channel {channel_id}: {e}")
                return
        try:
            await channel.send(content)
            log.info(f"Sent message to channel {channel_id}: {content}")
        except Exception as e:
            log.error(f"Failed to send message to channel {channel_id}: {e}")

    async def route_sweep(self, sweep_data: List[Dict]) -> Dict:
        # Load memos from bot_data/memos.jsonl
        memos = []
        memos_path = "bot_data/memos.jsonl"
        if os.path.exists(memos_path):
            with open(memos_path, "r") as f:
                for line in f:
                    try:
                        memos.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        log.warning(f"Skipping invalid memo line: {line.strip()}")
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a bot that reviews recent messages in a Discord channel and decides what actions to take. "
                    "You can choose to do nothing (noop), send a message to a channel, save a memo, or execute a custom command. "
                    "You are allowed up to {max_actions} actions per sweep, and up to {max_iterations} sweeps (iterations) per run. "
                    "This is iteration 1 out of {max_iterations}. "
                    "You have {max_actions} actions available in this sweep (including this run). "
                    "Analyze the provided messages and determine the appropriate actions based on their content."
                ).format(
                    max_actions=self.max_actions_per_sweep,
                    max_iterations=self.max_iterations
                )
            },
            {   "role": "user", "content": "Prepare for sweep." },
            {   "role": "assistant", "content": f"Loading memos..." },
            {   "role": "assistant", "content": f"Loaded {len(memos)} memos." },
            {
                "role": "assistant",
                "content": (
                    f"Here are the memos that have saved so far:\n{json.dumps(memos, indent=2)}\n\n"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here are the recent messages:\n{str(sweep_data)}\n\n"
                    "Based on these messages, decide what actions to take. "
                    "Return your decision in JSON format following this schema:\n"
                    f"{str(router_decision_schema)}"
                )
            },
            { "role": "system", "content": "Internal commands available: send_message (requires channel_id and message), save_memo (requires memo), execute_custom_command (requires command and payload). Noop will exit the loop, and will skip any actions that wuold have otherwise followed it. The system may execute multiple actions in one iteration, up to the max allowed, or may execute as few as it deems necessary, with at least one. Alternatively, the system may execute some actions, and wait for the results before decisiong what to do next." },
            # Rundown on available commands could go here
            #{"role": "system", "content": "Available custom commands: []"}
        ]

        actions_spent = 0
        current_iteration = 1
        for iteration in range(self.max_iterations):
            print(f"--- Iteration {current_iteration} ---")
            print(f"Actions spent so far: {actions_spent}")
            prompt.append({
                "role": "assistant",
                "content": f"Current iteration {current_iteration} out of {self.max_iterations}. We have {self.max_actions_per_sweep - actions_spent} actions remaining. "
            })
            prompt.append({
                "role": "user",
                "content": "Make next decision chain. Be reasonably social. You don't have to respond all the time, but make sure you're actually responding to users who are trying to engage with you. Use proper described JSON format."
            })
            response = client.chat.completions.create(
                model=self.router_model,
                messages=prompt,
                max_completion_tokens=1000
            )

            decision = response.choices[0].message.content
            print(f"LLM Decision: {decision}")
            try:
                decision_json = json.loads(decision)
                # Basic validation of the response structure
                if ("actions" in decision_json and
                    isinstance(decision_json["actions"], list)):
                    for action in decision_json["actions"]:
                        if action["type"] == "send_message":
                            await self.send_message(action["channel_id"], action["message"])
                            prompt.append({
                                "role": "assistant",
                                "content": f"Sent message to {action['channel_id']}: {action['message']}"
                            })
                        elif action["type"] == "save_memo":
                            save_memo(action["memo"])
                            prompt.append({
                                "role": "assistant",
                                "content": f"Saved memo: {action['memo']}"
                            })
                        elif action["type"] == "execute_command":
                            result = await execute_custom_command(action["name"], action["payload"])    
                            prompt.append({
                                "role": "assistant",
                                "content": f"Executed command {action['name']} with payload {action['payload']}. Result: {result}"
                            })
                        elif action["type"] == "noop":
                            log.info("Noop action received. Exiting loop.")
                            return
                        else:
                            prompt.append({
                                "role": "assistant",
                                "content": f"Unknown action type: {action['type']}"
                            })
                            log.warning(f"Unknown action type: {action['type']}")

                    # Update the actions spent and remaining actions
                    actions_spent += len(decision_json["actions"])
                    current_iteration += 1
                    if actions_spent >= self.max_actions_per_sweep:
                        log.info("Max actions spent for this sweep. Exiting loop.")
                        return
                    
            # Determine if the response is valid JSON and matches the schema
            except json.JSONDecodeError:
                log.error("LLM response is not valid JSON.")
                prompt.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content
                })


    @tasks.loop(seconds=30)
    async def sweep_recent_activity(self):
        """Periodically sweep monitored channels for new messages."""
        now = discord.utils.utcnow()

        for channel_id in self.monitored_channel_ids:
            try:
                channel = self.get_channel(channel_id) or await self.fetch_channel(channel_id)

                # Only text-y places (TextChannel or Thread)
                if not isinstance(channel, (discord.TextChannel, discord.Thread)):
                    log.info(f"Skipping non-text channel {channel_id}")
                    continue

                after_ts = self._last_seen.get(channel_id, now - self.startup_lookback)
                newest_ts = after_ts
                count = 0

                # oldest_first=True ensures stable checkpointing
                sweep = Sweep(channel_id=channel_id, after=after_ts)
                async for message in channel.history(
                    limit=self.max_messages_per_channel,
                    after=after_ts,
                    oldest_first=True
                ):
                    count += 1
                    if message.created_at > newest_ts:
                        sweep.add_message(message)
                        newest_ts = message.created_at

                if count > 0:
                    llm_data = sweep.to_llm_readable()
                    await self.route_sweep(llm_data)
                    self._last_seen[channel_id] = newest_ts
                    log.info(f"[#{channel.name}] swept {count} new messages since {after_ts.isoformat()}")

                # advance checkpoint even if no messages appeared
                self._last_seen[channel_id] = newest_ts
                log.info(f"[#{channel.name}] swept {count} new messages since {after_ts.isoformat()}")

            except Exception:
                log.exception(f"Error while sweeping channel {channel_id}")

    @sweep_recent_activity.before_loop
    async def before_sweep(self):
        await self.wait_until_ready()
        log.info("Starting sweep loop…")


def parse_channel_ids(env_value: str) -> List[int]:
    parts = [p.strip() for p in env_value.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    token = os.environ["DISCORD_TOKEN"]

    # Supply channel IDs via env or hardcode below.
    env_ids = os.getenv("MONITORED_CHANNEL_IDS", "")
    monitored_ids = parse_channel_ids(env_ids) if env_ids else [
        1410007094144405635
    ]

    intents = discord.Intents.default()
    intents.guilds = True
    intents.messages = True
    intents.message_content = True  # enable in Developer Portal for reading message content/history

    bot = SweepBot(
        bot_name="Seshat",
        monitored_channel_ids=monitored_ids,
        sweep_interval_seconds=300,        # every 30 seconds
        startup_lookback_minutes=60,       # first sweep looks back 60 minutes
        max_messages_per_channel=500,      # cap per sweep
        router_model="gpt-5-mini",
        max_iterations=2,
        max_actions_per_sweep=3,
        intents=intents,
    )

    bot.run(token)


if __name__ == "__main__":
    main()
