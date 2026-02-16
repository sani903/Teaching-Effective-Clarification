import os
from uuid import uuid4

import pytest

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream
from openhands.events.action import AgentFinishAction, AgentRejectAction, MessageAction
from openhands.llm.llm import LLM
from openhands.storage.memory import InMemoryFileStore


# Register our test agent
class TestAgent(Agent):
    def step(self, state):
        return MessageAction(content='Test agent response')


Agent.register('test_agent', TestAgent)


class TestRealPrePostConditions:
    """Test pre/postconditions with real LLM calls."""

    @pytest.fixture
    def event_stream(self):
        """Create a real event stream for testing."""
        file_store = InMemoryFileStore()
        return EventStream(str(uuid4()), file_store)

    @pytest.fixture
    def real_agent(self):
        """Create a real agent with a real LLM."""
        # Create a minimal config for a small, fast LLM
        # Using either Claude or OpenAI's smallest models based on what's available
        llm_config = LLMConfig(
            model=os.environ.get('TEST_LLM_MODEL', 'neulab/claude-3-5-haiku-20241022'),
            temperature=0.0,
            max_message_chars=8000,
        )
        agent_config = AgentConfig()

        # Initialize a real LLM
        llm = LLM(config=llm_config)

        # Create an agent with the real LLM
        agent = TestAgent(llm=llm, config=agent_config)
        return agent

    @pytest.fixture
    def controller(self, event_stream, real_agent):
        """Create a controller with real models."""
        # Create the controller with real models
        controller = AgentController(
            agent=real_agent,
            event_stream=event_stream,
            max_iterations=5,
            sid=str(uuid4()),
            initial_state=State(session_id=str(uuid4())),
            postconditions_model=os.environ.get(
                'TEST_LLM_MODEL', 'neulab/claude-3-5-haiku-20241022'
            ),
            preconditions_model=os.environ.get(
                'TEST_LLM_MODEL', 'neulab/claude-3-5-haiku-20241022'
            ),
        )

        # Initialize required properties
        controller._first_user_message_processed = False
        controller.to_refine = False

        # Override set_agent_state_to to avoid actual state changes
        async def mock_set_state(state):
            controller.state.agent_state = state

        controller.set_agent_state_to = mock_set_state

        yield controller

    @pytest.mark.asyncio
    async def test_preconditions_real_generation(self, controller):
        """Test real generation of preconditions for a task."""
        # Create a simple user message
        user_message = MessageAction(
            content='Create a Python script that sorts a list of numbers.'
        )
        user_message._source = EventSource.USER

        # Process the message to generate preconditions
        await controller._handle_message_action(user_message)

        # Verify preconditions were generated
        assert controller._first_user_message_processed is True
        assert controller.state.preconditions is not None
        assert len(controller.state.preconditions) > 0
        assert (
            controller.initial_task
            == 'Create a Python script that sorts a list of numbers.'
        )

        # Check augmented message format
        assert 'Generated Checklist' in user_message.content
        assert (
            'Create a Python script that sorts a list of numbers.'
            in user_message.content
        )

        # Print the generated preconditions for inspection
        print(f'\nPreconditions generated: {controller.state.preconditions}')

    @pytest.mark.asyncio
    async def test_postconditions_real_generation(self, controller):
        """Test real generation of postconditions for a task."""
        # Set up necessary state
        controller.initial_task = 'Create a Python script that sorts a list of numbers.'
        controller.postconditions_passed = False

        # Add some events to create a trajectory
        controller.event_stream.add_event(
            MessageAction(
                content='Create a Python script that sorts a list of numbers.'
            ),
            EventSource.USER,
        )
        controller.event_stream.add_event(
            MessageAction(
                content="""
def sort_numbers(numbers):
    return sorted(numbers)

# Example usage
if __name__ == "__main__":
    my_list = [5, 2, 8, 1, 9, 3]
    sorted_list = sort_numbers(my_list)
    print(f"Original list: {my_list}")
    print(f"Sorted list: {sorted_list}")
"""
            ),
            EventSource.AGENT,
        )

        # Create finish action to trigger postconditions
        finish_action = AgentFinishAction(outputs={'result': 'Task completed'})

        # Process the action
        await controller._handle_action(finish_action)

        # Verify postconditions were generated
        assert controller.postconditions_passed is True
        assert controller.state.postconditions is not None
        assert len(controller.state.postconditions) > 0

        # Print the generated postconditions for inspection
        print(f'\nPostconditions generated: {controller.state.postconditions}')

        # Check that verification message was added to event stream
        events = list(controller.event_stream.get_events())
        verification_msg = [
            e
            for e in events
            if isinstance(e, MessageAction)
            and 'Check how many of the items' in e.content
        ]
        assert len(verification_msg) > 0

    @pytest.mark.asyncio
    async def test_refinement_mode(self, event_stream, real_agent):
        """Test refinement mode behavior with real models."""
        # Create a controller with refinement enabled
        controller = AgentController(
            agent=real_agent,
            event_stream=event_stream,
            max_iterations=5,
            sid=str(uuid4()),
            initial_state=State(session_id=str(uuid4())),
            postconditions_model=os.environ.get(
                'TEST_LLM_MODEL', 'neulab/claude-3-5-haiku-20241022'
            ),
            preconditions_model=os.environ.get(
                'TEST_LLM_MODEL', 'neulab/claude-3-5-haiku-20241022'
            ),
            to_refine=True,
        )

        # Set up required state
        controller.initial_task = 'Create a Python script that sorts a list of numbers.'
        controller.postconditions_passed = False
        controller._initial_max_iterations = 5

        # Override set_agent_state_to
        async def mock_set_state(state):
            controller.state.agent_state = state

        controller.set_agent_state_to = mock_set_state

        # Add some events to create a trajectory
        controller.event_stream.add_event(
            MessageAction(
                content='Create a Python script that sorts a list of numbers.'
            ),
            EventSource.USER,
        )
        controller.event_stream.add_event(
            MessageAction(
                content="""
def sort_numbers(numbers):
    return sorted(numbers)

# No example usage provided
"""
            ),
            EventSource.AGENT,
        )

        # Trigger postconditions with finish action
        finish_action = AgentFinishAction(outputs={'result': 'Task completed'})
        await controller._handle_action(finish_action)

        # Verify refinement-specific behavior
        assert controller.postconditions_passed is True
        assert controller.state.postconditions is not None

        # Find refinement message in events
        events = list(controller.event_stream.get_events())
        refinement_msg = [
            e
            for e in events
            if isinstance(e, MessageAction) and 'refine your solution' in e.content
        ]
        assert len(refinement_msg) > 0

        # Verify max iterations was extended
        assert controller.state.max_iterations == controller.state.iteration + 5

    @pytest.mark.asyncio
    async def test_edge_case_empty_task(self, controller):
        """Test behavior with empty task string."""
        # Create an empty user message
        empty_message = MessageAction(content='')
        empty_message._source = EventSource.USER

        # Process the message
        await controller._handle_message_action(empty_message)

        # Verify handling of empty task
        assert controller._first_user_message_processed is True
        assert controller.initial_task == ''

        # Empty task should still generate some preconditions (model-dependent)
        # This might be a generic checklist or an error message
        assert controller.state.preconditions is not None

        print(f'\nPreconditions for empty task: {controller.state.preconditions}')

    @pytest.mark.asyncio
    async def test_multiple_iterations_stuck_detection(self, controller):
        """Test stuck detection and postconditions generation."""
        # Set up state for stuck detection
        controller.initial_task = 'Create a Python function to calculate factorial.'
        controller.postconditions_passed = False
        controller.state.iteration = 3
        controller.state.agent_state = AgentState.RUNNING

        # Add some trajectory events
        controller.event_stream.add_event(
            MessageAction(content='Create a Python function to calculate factorial.'),
            EventSource.USER,
        )
        controller.event_stream.add_event(
            MessageAction(content="I'll write a factorial function."), EventSource.AGENT
        )

        # Simulate being stuck by making the detector return True
        controller._is_stuck = lambda: True

        # Call _step to trigger stuck handling
        await controller._step()

        # Verify postconditions were generated for stuck detection
        assert controller.postconditions_passed is True
        assert controller.state.postconditions is not None

        # Verify events were added to continue execution
        events = list(controller.event_stream.get_events())
        checklist_msg = [
            e
            for e in events
            if isinstance(e, MessageAction) and 'CHECKLIST' in e.content
        ]
        assert len(checklist_msg) > 0

    @pytest.mark.asyncio
    async def test_reject_action_handling(self, controller):
        """Test handling of reject action."""
        # Set up necessary state
        controller.initial_task = 'Create a Python script that sorts a list of numbers.'
        controller.postconditions_passed = False

        # Create a reject action (agent decides it can't complete the task)
        reject_action = AgentRejectAction(outputs={'reason': 'Task is ambiguous'})

        # Process the action
        await controller._handle_action(reject_action)

        # Verify the agent state was set to REJECTED
        assert controller.state.agent_state == AgentState.REJECTED

        # No postconditions should be generated for rejection
        assert controller.postconditions_passed is False
        assert controller.state.postconditions is None

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, controller):
        """Test behavior when max iterations is reached."""
        # Set up state to trigger max iterations
        controller.initial_task = 'Create a Python function to calculate factorial.'
        controller.postconditions_passed = False
        controller.state.iteration = 5
        controller.state.max_iterations = 5
        controller.state.agent_state = AgentState.RUNNING

        # Add some trajectory events
        controller.event_stream.add_event(
            MessageAction(content='Create a Python function to calculate factorial.'),
            EventSource.USER,
        )
        controller.event_stream.add_event(
            MessageAction(content="Here's a factorial function:"), EventSource.AGENT
        )

        # Mock traffic control to return False (continue execution)
        async def mock_traffic_control(*args):
            return False

        controller._handle_traffic_control = mock_traffic_control

        # Make sure we don't detect being stuck
        controller._is_stuck = lambda: False

        # Call _step to trigger max iterations handling
        await controller._step()

        # Verify postconditions were generated
        assert controller.postconditions_passed is True
        assert controller.state.postconditions is not None

        # Check for max iterations message
        events = list(controller.event_stream.get_events())
        max_iter_msg = [
            e
            for e in events
            if isinstance(e, MessageAction) and 'maximum number of steps' in e.content
        ]
        assert len(max_iter_msg) > 0

        # Verify buffer iterations were added
        assert controller.state.max_iterations == 5 + 2  # Original + buffer

    @pytest.mark.asyncio
    async def test_unicode_special_chars(self, controller):
        """Test with Unicode and special characters."""
        # Create a message with various Unicode and special characters
        special_content = '🚀 Testing with Unicode: ñáéíóú 汉字 עִבְרִית ✨ \u200b\t\n'
        user_message = MessageAction(content=special_content)
        user_message._source = EventSource.USER

        # Process the message
        await controller._handle_message_action(user_message)

        # Verify handling of special characters
        assert controller._first_user_message_processed is True
        assert controller.initial_task == special_content

        # Checklist should be generated without errors
        assert controller.state.preconditions is not None

        # Verify Unicode preserved in augmented message
        assert '🚀' in user_message.content
        assert 'ñáéíóú' in user_message.content
        assert '汉字' in user_message.content

    @pytest.mark.asyncio
    async def test_very_large_prompt(self, controller):
        """Test with a very large prompt."""
        # Create a large message (5KB)
        large_content = 'Task: Create a Python function. ' + (
            'Details: This is part of the requirements. ' * 100
        )
        user_message = MessageAction(content=large_content)
        user_message._source = EventSource.USER

        # Process the message
        await controller._handle_message_action(user_message)

        # Verify large content handling
        assert controller._first_user_message_processed is True
        assert controller.initial_task == large_content

        # Checklist should be generated for large input
        assert controller.state.preconditions is not None

        # Check that the augmented message contains both original content and checklist
        assert 'Task: Create a Python function' in user_message.content
        assert 'Generated Checklist' in user_message.content
