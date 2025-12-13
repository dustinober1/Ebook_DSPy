"""
Intelligent Agents Implementation Examples

This file demonstrates various types of intelligent agents built with DSPy,
including reactive, proactive, collaborative, and learning agents.

Examples include:
- Basic reactive agents
- Planning and goal-oriented agents
- Collaborative multi-agent systems
- Learning and adaptive agents
- Real-world agent applications
"""

import dspy
from dspy.teleprompter import BootstrapFewShot, MIPRO
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Configure language model (placeholder)
dspy.settings.configure(lm=dspy.LM(model="gpt-3.5-turbo", api_key="your-key"))

# Example 1: Basic Reactive Agent
class ReactiveAgent(dspy.Module):
    """Simple reactive agent that responds to inputs."""

    def __init__(self, name, capabilities):
        super().__init__()
        self.name = name
        self.capabilities = capabilities
        self.perceive = dspy.Predict("input -> perceived_state")
        self.decide = dspy.Predict("state, capabilities -> action, reasoning")
        self.memory = []

    def forward(self, input_text):
        # Perceive the current state
        perception = self.perceive(input=input_text)
        current_state = perception.perceived_state

        # Decide on action based on state
        decision = self.decide(
            state=current_state,
            capabilities=", ".join(self.capabilities)
        )

        # Store in memory
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "state": current_state,
            "action": decision.action,
            "reasoning": decision.reasoning
        })

        return dspy.Prediction(
            agent_name=self.name,
            perceived_state=current_state,
            action=decision.action,
            reasoning=decision.reasoning
        )

def demo_reactive_agent():
    """Demonstrate reactive agent behavior."""
    print("=" * 60)
    print("Example 1: Basic Reactive Agent")
    print("=" * 60)

    # Create a customer service agent
    agent = ReactiveAgent(
        name="SupportBot",
        capabilities=["answer_questions", "escalate_issues", "provide_resources"]
    )

    # Simulate customer interactions
    interactions = [
        "I need help with my order",
        "The website is down",
        "I want to return my purchase"
    ]

    print("\nCustomer Interactions:")
    for interaction in interactions:
        print(f"\nCustomer: {interaction}")
        response = agent(interaction)
        print(f"Agent Action: {response.action}")
        print(f"Reasoning: {response.reasoning}")

    print(f"\nTotal interactions: {len(agent.memory)}")

# Example 2: Goal-Oriented Planning Agent
class PlanningAgent(dspy.Module):
    """Agent that plans and executes to achieve goals."""

    def __init__(self, name, goals):
        super().__init__()
        self.name = name
        self.goals = goals
        self.analyze_goal = dspy.Predict("goal, context -> requirements, constraints")
        self.create_plan = dspy.ChainOfThought("requirements, constraints -> plan, steps")
        self.execute_step = dspy.Predict("plan, current_step -> action, next_step")
        self.evaluate_progress = dspy.Predict("goal, current_state -> progress, adjustments")

        self.current_plan = None
        self.current_step = 0
        self.progress_history = []

    def forward(self, goal, context=None):
        # Analyze the goal
        analysis = self.analyze_goal(
            goal=goal,
            context=context or "General context"
        )

        # Create or update plan
        if not self.current_plan:
            planning = self.create_plan(
                requirements=analysis.requirements,
                constraints=analysis.constraints
            )
            self.current_plan = planning.plan
            self.current_step = 0

        # Execute current step
        execution = self.execute_step(
            plan=self.current_plan,
            current_step=str(self.current_step)
        )

        # Evaluate progress
        progress = self.evaluate_progress(
            goal=goal,
            current_state=f"Step {self.current_step} completed"
        )

        # Update progress
        self.progress_history.append({
            "step": self.current_step,
            "progress": progress.progress,
            "action": execution.action
        })

        # Update step
        if execution.next_step:
            self.current_step = int(execution.next_step)

        return dspy.Prediction(
            goal=goal,
            plan=self.current_plan,
            current_action=execution.action,
            progress=progress.progress,
            adjustments=progress.adjustments,
            steps_completed=self.current_step
        )

def demo_planning_agent():
    """Demonstrate planning agent behavior."""
    print("\n" + "=" * 60)
    print("Example 2: Goal-Oriented Planning Agent")
    print("=" * 60)

    agent = PlanningAgent(
        name="ProjectManager",
        goals=["Deliver project on time", "Ensure quality", "Manage resources"]
    )

    # Work on a project goal
    goal = "Launch new e-commerce website in 3 months"
    context = "Team of 5 developers, limited budget"

    print(f"\nGoal: {goal}")
    print(f"Context: {context}")

    # Simulate plan execution
    for step in range(4):
        print(f"\n--- Step {step + 1} ---")
        result = agent(goal=goal, context=context)
        print(f"Action: {result.current_action}")
        print(f"Progress: {result.progress}")
        if result.adjustments:
            print(f"Adjustments: {result.adjustments}")

# Example 3: Collaborative Multi-Agent System
class CollaborativeAgent(dspy.Module):
    """Agent that can collaborate with other agents."""

    def __init__(self, name, expertise, team=None):
        super().__init__()
        self.name = name
        self.expertise = expertise
        self.team = team or []
        self.analyze_task = dspy.Predict("task -> complexity, required_skills")
        self.delegate_or_handle = dspy.Predict(
            "task, expertise, team_skills -> decision, delegation_details"
        )
        self.collaborate = dspy.Predict("agent, task -> collaboration_message")
        self.integrate_results = dspy.ChainOfThought(
            "task, individual_results -> integrated_solution"
        )

    def forward(self, task):
        # Analyze task requirements
        analysis = self.analyze_task(task=task)

        # Get team skills
        team_skills = [(agent["name"], agent["expertise"]) for agent in self.team]

        # Decide whether to handle or delegate
        decision = self.delegate_or_handle(
            task=task,
            expertise=self.expertise,
            team_skills=str(team_skills)
        )

        results = []

        if decision.decision == "handle_myself":
            # Handle task personally
            results.append({
                "agent": self.name,
                "contribution": f"Handled {task} using {self.expertise} expertise"
            })
        else:
            # Delegate to appropriate team member
            for member in self.team:
                if member["expertise"] in decision.delegation_details:
                    # Send collaboration message
                    collab = self.collaborate(
                        agent=member["name"],
                        task=task
                    )
                    results.append({
                        "agent": member["name"],
                        "contribution": collab.collaboration_message
                    })

        # Integrate results
        integration = self.integrate_results(
            task=task,
            individual_results=str(results)
        )

        return dspy.Prediction(
            task=task,
            task_complexity=analysis.complexity,
            decision=decision.decision,
            contributions=results,
            integrated_solution=integration.integrated_solution
        )

def demo_collaborative_agents():
    """Demonstrate collaborative agent system."""
    print("\n" + "=" * 60)
    print("Example 3: Collaborative Multi-Agent System")
    print("=" * 60)

    # Create a team
    team = [
        {"name": "UIAgent", "expertise": "User Interface Design"},
        {"name": "BackendAgent", "expertise": "Server Development"},
        {"name": "DataAgent", "expertise": "Data Analysis"},
        {"name": "SecurityAgent", "expertise": "Security"}
    ]

    # Lead agent
    lead_agent = CollaborativeAgent(
        name="TechLead",
        expertise="System Architecture",
        team=team
    )

    # Complex project task
    task = "Build a secure user authentication system"

    print(f"\nProject Task: {task}")
    print(f"Team Members: {[a['name'] for a in team]}")

    result = lead_agent(task=task)

    print(f"\nTask Complexity: {result.task_complexity}")
    print(f"Decision: {result.decision}")
    print("\nTeam Contributions:")
    for contribution in result.contributions:
        print(f"- {contribution['agent']}: {contribution['contribution']}")
    print(f"\nIntegrated Solution: {result.integrated_solution}")

# Example 4: Learning Adaptive Agent
class LearningAgent(dspy.Module):
    """Agent that learns from experience and improves over time."""

    def __init__(self, name, learning_domain):
        super().__init__()
        self.name = name
        self.learning_domain = learning_domain
        self.process_input = dspy.Predict("input, history -> processed_input, patterns")
        self.generate_response = dspy.ChainOfThought(
            "processed_input, learned_patterns -> response, confidence"
        )
        self.learn_from_feedback = dspy.Predict(
            "action, outcome, feedback -> learning_insight, strategy_update"
        )

        # Learning components
        self.experiences = []
        self.patterns = []
        self.strategies = {}
        self.performance_metrics = []

    def forward(self, input_text, feedback=None):
        # Process input with learned patterns
        patterns_text = "\n".join(self.patterns[-5:])  # Recent patterns
        processing = self.process_input(
            input=input_text,
            history=patterns_text
        )

        # Generate response
        response = self.generate_response(
            processed_input=processing.processed_input,
            learned_patterns=patterns_text
        )

        # Store experience
        experience = {
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "processed": processing.processed_input,
            "response": response.response,
            "confidence": response.confidence,
            "patterns": processing.patterns
        }

        self.experiences.append(experience)

        # Learn from feedback if available
        if feedback:
            learning = self.learn_from_feedback(
                action=response.response,
                outcome="processed" in feedback.lower(),
                feedback=feedback
            )

            # Update patterns and strategies
            if learning.learning_insight:
                self.patterns.append(learning.learning_insight)

            if learning.strategy_update:
                self.strategies[processing.processed_input[:50]] = learning.strategy_update

            # Update performance
            performance = 1.0 if "good" in feedback.lower() else 0.5
            self.performance_metrics.append(performance)

            experience["feedback"] = feedback
            experience["learning"] = learning.learning_insight

        return dspy.Prediction(
            response=response.response,
            confidence=response.confidence,
            patterns_detected=processing.patterns,
            total_experiences=len(self.experiences),
            current_performance=sum(self.performance_metrics[-5:]) / min(5, len(self.performance_metrics)) if self.performance_metrics else 0
        )

    def get_learning_summary(self):
        """Get summary of learning progress."""
        avg_performance = sum(self.performance_metrics) / len(self.performance_metrics) if self.performance_metrics else 0
        return {
            "total_experiences": len(self.experiences),
            "patterns_learned": len(self.patterns),
            "strategies_developed": len(self.strategies),
            "average_performance": avg_performance,
            "recent_performance": self.performance_metrics[-10:] if self.performance_metrics else []
        }

def demo_learning_agent():
    """Demonstrate learning agent behavior."""
    print("\n" + "=" * 60)
    print("Example 4: Learning Adaptive Agent")
    print("=" * 60)

    # Create a learning agent for customer service
    agent = LearningAgent(
        name="SupportLearner",
        learning_domain="Customer Support"
    )

    # Simulate learning interactions
    interactions = [
        ("I need help with my order", "Good response, helpful"),
        ("The website is not working", "Could be more specific"),
        ("How do I reset my password?", "Perfect, step-by-step guide"),
        ("What are your business hours?", "Clear and concise"),
        ("I want to speak to a human", "Good escalation")
    ]

    print("\nLearning Interactions:")
    for input_text, feedback in interactions:
        print(f"\nCustomer: {input_text}")
        result = agent(input_text=input_text, feedback=feedback)
        print(f"Agent: {result.response}")
        print(f"Confidence: {result.confidence}")
        print(f"Current Performance: {result.current_performance:.2f}")

    # Show learning summary
    summary = agent.get_learning_summary()
    print(f"\nLearning Summary:")
    print(f"- Total Experiences: {summary['total_experiences']}")
    print(f"- Patterns Learned: {summary['patterns_learned']}")
    print(f"- Strategies Developed: {summary['strategies_developed']}")
    print(f"- Average Performance: {summary['average_performance']:.2f}")

# Example 5: Real-World Customer Service Agent
class CustomerServiceAgent(dspy.Module):
    """Production-ready customer service agent."""

    def __init__(self, company_name, knowledge_base):
        super().__init__()
        self.company_name = company_name
        self.knowledge_base = knowledge_base

        self.classify_intent = dspy.Predict(
            "customer_message -> intent, sentiment, urgency, complexity"
        )
        self.find_solution = dspy.ChainOfThought(
            "intent, sentiment, knowledge_base -> solution, empathy_level"
        )
        self.escalate_check = dspy.Predict(
            "solution, sentiment, complexity -> should_escalate, escalation_reason"
        )
        self.personalize_response = dspy.Predict(
            "solution, customer_profile -> personalized_response, follow_up_actions"
        )

        # Session management
        self.sessions = {}

    def forward(self, customer_message, session_id=None, customer_profile=None):
        # Get or create session
        if not session_id:
            session_id = f"session_{len(self.sessions)}"
        session = self.sessions.get(session_id, {"messages": [], "resolved": False})

        # Classify the message
        classification = self.classify_intent(customer_message=customer_message)

        # Find solution based on intent
        solution = self.find_solution(
            intent=classification.intent,
            sentiment=classification.sentiment,
            knowledge_base=self.knowledge_base
        )

        # Check if escalation is needed
        escalation = self.escalate_check(
            solution=solution.solution,
            sentiment=classification.sentiment,
            complexity=classification.complexity
        )

        # Personalize response
        personalization = self.personalize_response(
            solution=solution.solution,
            customer_profile=customer_profile or "New customer"
        )

        # Update session
        session["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "customer_message": customer_message,
            "agent_response": personalization.personalized_response,
            "intent": classification.intent,
            "escalated": escalation.should_escalate == "yes"
        })
        self.sessions[session_id] = session

        return dspy.Prediction(
            session_id=session_id,
            intent=classification.intent,
            sentiment=classification.sentiment,
            urgency=classification.urgency,
            response=personalization.personalized_response,
            follow_up_actions=personalization.follow_up_actions,
            escalated=escalation.should_escalate == "yes",
            escalation_reason=escalation.escalation_reason,
            empathy_level=solution.empathy_level
        )

def demo_customer_service_agent():
    """Demonstrate production customer service agent."""
    print("\n" + "=" * 60)
    print("Example 5: Production Customer Service Agent")
    print("=" * 60)

    # Knowledge base
    kb = """
    Return Policy: 30-day return policy with full refund.
    Shipping: Free shipping on orders over $50.
    Support Hours: 24/7 chat, 9-5 phone support.
    Payment Methods: Credit card, PayPal, Apple Pay.
    """

    agent = CustomerServiceAgent("TechStore", kb)

    # Customer interactions
    interactions = [
        ("I want to return my purchase", {"tier": "premium", "order_count": 5}),
        ("How long does shipping take?", {"tier": "standard", "order_count": 1}),
        ("My package arrived damaged!", {"tier": "vip", "order_count": 20})
    ]

    print("\nCustomer Service Interactions:")
    for message, profile in interactions:
        print(f"\nCustomer ({profile['tier']}): {message}")
        result = agent(customer_message=message, customer_profile=profile)

        print(f"Intent: {result.intent}")
        print(f"Urgency: {result.urgency}")
        print(f"Response: {result.response}")
        print(f"Empathy Level: {result.empathy_level}")
        if result.escalated:
            print(f"ESCALATED: {result.escalation_reason}")
        if result.follow_up_actions:
            print(f"Follow-up: {result.follow_up_actions}")

    # Session summary
    print(f"\nTotal Sessions: {len(agent.sessions)}")
    for sid, session in agent.sessions.items():
        escalated_count = sum(1 for m in session["messages"] if m["escalated"])
        print(f"- {sid}: {len(session['messages'])} messages, {escalated_count} escalated")

# Main execution
def run_all_examples():
    """Run all intelligent agent examples."""
    print("DSPy Intelligent Agents Examples")
    print("Demonstrating various agent types and capabilities\n")

    try:
        demo_reactive_agent()
        demo_planning_agent()
        demo_collaborative_agents()
        demo_learning_agent()
        demo_customer_service_agent()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All intelligent agent examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_examples()