from datetime import date

import pytest

from agentdesk import db
from agentdesk.agent import Agent
from agentdesk.config import Settings
from agentdesk.guardrails import Guard
from agentdesk.knowledge import PolicyKnowledgeBase
from agentdesk.llm import NaivePlanner
from agentdesk.tools import Toolbox
from agentdesk.tracing import Tracer

TODAY = date(2026, 11, 5)


@pytest.fixture()
def settings():
    return Settings(today=TODAY)


@pytest.fixture()
def conn(settings):
    connection = db.connect()
    db.init_db(connection, settings.today)
    return connection


@pytest.fixture(scope="session")
def kb():
    return PolicyKnowledgeBase()


@pytest.fixture()
def toolbox(conn, kb, settings):
    return Toolbox(conn, kb, settings)


@pytest.fixture()
def guard(conn, settings, toolbox):
    return Guard(conn, settings, True, toolbox.names)


@pytest.fixture()
def agent(toolbox, guard, settings):
    return Agent(NaivePlanner(), toolbox, guard, Tracer(), settings)
