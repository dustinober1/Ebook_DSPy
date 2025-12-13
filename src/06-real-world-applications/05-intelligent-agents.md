# Intelligent Agents: Building Autonomous Problem-Solving Systems

## Introduction

Intelligent agents represent one of the most exciting applications of language models. These are autonomous systems that can perceive their environment, reason about problems, make decisions, and take actions to achieve specific goals. From customer service bots to research assistants, intelligent agents are transforming how we interact with and leverage AI in real-world scenarios.

## Understanding Intelligent Agents

### Core Components of an Agent

1. **Perception**: Understanding the current state and environment
2. **Planning**: Developing strategies to achieve goals
3. **Decision Making**: Choosing the best course of action
4. **Execution**: Carrying out planned actions
5. **Learning**: Improving from experience and feedback
6. **Memory**: Maintaining context and knowledge over time

### Agent Types

- **Reactive Agents**: Respond directly to current inputs
- **Proactive Agents**: Anticipate future needs and take initiative
- **Social Agents**: Understand and respond to human emotions and social cues
- **Collaborative Agents**: Work with other agents or humans
- **Learning Agents**: Improve performance over time

## Building Intelligent Agents with DSPy

### Basic Reactive Agent

```python
import dspy
from typing import List, Dict, Any, Optional

class ReactiveAgent(dspy.Module):
    def __init__(self, name, capabilities):
        super().__init__()
        self.name = name
        self.capabilities = capabilities
        self.perceive = dspy.Predict("input -> perceived_state")
        self.decide = dspy.Predict("state, capabilities -> action, reasoning")
        self.memory = {}

    def forward(self, input_text):
        # Perceive the current state
        perception = self.perceive(input=input_text)
        current_state = perception.perceived_state

        # Decide on action
        decision = self.decide(
            state=current_state,
            capabilities=", ".join(self.capabilities)
        )

        # Store in memory
        self.memory[len(self.memory)] = {
            "input": input_text,
            "state": current_state,
            "action": decision.action,
            "reasoning": decision.reasoning
        }

        return dspy.Prediction(
            agent_name=self.name,
            perceived_state=current_state,
            action=decision.action,
            reasoning=decision.reasoning
        )

    def get_memory(self, num_recent=5):
        """Get recent memory entries."""
        return dict(list(self.memory.items())[-num_recent:])
```

### Proactive Agent with Planning

```python
class ProactiveAgent(dspy.Module):
    def __init__(self, name, goals, tools):
        super().__init__()
        self.name = name
        self.goals = goals
        self.tools = tools
        self.understand_context = dspy.Predict("input -> context, user_intent")
        self.create_plan = dspy.ChainOfThought("context, intent, goals, tools -> plan")
        self.execute_step = dspy.Predict("plan, current_step, tools -> action, next_step")
        self.evaluate_progress = dspy.Predict("goal, current_state -> progress_score, next_objective")

        # Persistent memory
        self.memory = []
        self.current_plan = None
        self.current_step = 0

    def forward(self, input_text):
        # Understand context and intent
        understanding = self.understand_context(input=input_text)
        context = understanding.context
        intent = understanding.user_intent

        # Create or update plan
        if not self.current_plan or self._needs_new_plan(intent):
            planning = self.create_plan(
                context=context,
                intent=intent,
                goals=", ".join(self.goals),
                tools=", ".join(self.tools)
            )
            self.current_plan = planning.plan
            self.current_step = 0

        # Execute current step
        execution = self.execute_step(
            plan=self.current_plan,
            current_step=str(self.current_step),
            tools=", ".join(self.tools)
        )

        # Evaluate progress
        progress = self.evaluate_progress(
            goal=self.goals[0],  # Primary goal
            current_state=context
        )

        # Update memory
        self.memory.append({
            "timestamp": len(self.memory),
            "input": input_text,
            "intent": intent,
            "action_taken": execution.action,
            "next_step": execution.next_step,
            "progress": progress.progress_score
        })

        # Update step
        if execution.next_step:
            self.current_step = int(execution.next_step)

        return dspy.Prediction(
            agent_name=self.name,
            user_intent=intent,
            action_taken=execution.action,
            current_plan=self.current_plan,
            progress=progress.progress_score,
            next_objective=progress.next_objective
        )

    def _needs_new_plan(self, intent):
        """Check if we need a new plan based on intent."""
        return "new" in intent.lower() or "different" in intent.lower()
```

### Collaborative Agent

```python
class CollaborativeAgent(dspy.Module):
    def __init__(self, name, expertise, team_members=None):
        super().__init__()
        self.name = name
        self.expertise = expertise
        self.team_members = team_members or []
        self.analyze_task = dspy.Predict("task -> task_type, complexity, requirements")
        self.delegate_or_handle = dspy.ChainOfThought(
            "task, expertise, team_members -> decision, rationale, delegation_details"
        )
        self.collaborate = dspy.Predict(
            "task, member_expertise -> collaboration_message"
        )
        self.synthesize_results = dspy.ChainOfThought(
            "task, individual_results -> final_result, confidence"
        )

        # Team communication
        self.messages = []
        self.shared_context = {}

    def forward(self, task):
        # Analyze the task
        analysis = self.analyze_task(task=task)

        # Decide whether to handle or delegate
        decision = self.delegate_or_handle(
            task=task,
            expertise=self.expertise,
            team_members=", ".join([m["name"] + ":" + m["expertise"] for m in self.team_members])
        )

        if decision.decision.lower() == "handle myself":
            # Handle the task personally
            result = self._handle_task(task)
        else:
            # Delegate to team member
            result = self._delegate_task(
                task,
                decision.delegation_details,
                decision.rationale
            )

        return dspy.Prediction(
            agent_name=self.name,
            task_analysis=analysis,
            decision=decision.decision,
            result=result,
            rationale=decision.rationale
        )

    def _handle_task(self, task):
        """Handle task personally based on expertise."""
        # Implementation would depend on specific expertise
        return f"Handled {task} using {self.expertise} expertise"

    def _delegate_task(self, task, delegation_details, rationale):
        """Delegate task to appropriate team member."""
        # Find appropriate team member
        for member in self.team_members:
            if member["expertise"].lower() in delegation_details.lower():
                # Send collaboration message
                collab_msg = self.collaborate(
                    task=task,
                    member_expertise=member["expertise"]
                )

                # Record collaboration
                self.messages.append({
                    "from": self.name,
                    "to": member["name"],
                    "message": collab_msg.collaboration_message,
                    "task": task
                })

                return f"Delegated {task} to {member['name']} (Expertise: {member['expertise']})"

        return f"No suitable team member found for: {task}"

    def add_team_member(self, name, expertise):
        """Add a new team member."""
        self.team_members.append({"name": name, "expertise": expertise})
```

### Learning Agent

```python
class LearningAgent(dspy.Module):
    def __init__(self, name, learning_goals):
        super().__init__()
        self.name = name
        self.learning_goals = learning_goals
        self.process_input = dspy.Predict("input -> processed_input, key_patterns")
        self.generate_response = dspy.ChainOfThought(
            "processed_input, past_experiences -> response, confidence"
        )
        self.evaluate_outcome = dspy.Predict("response, feedback -> learning_insight, strategy_update")
        self.reflect = dspy.ChainOfThought("experience, goal -> reflection, improvement_plan")

        # Learning components
        self.experiences = []  # Store past interactions
        self.patterns = {}     # Learned patterns
        self.strategies = {}   # Effective strategies
        self.performance_history = []

    def forward(self, input_text, feedback=None):
        # Process input
        processed = self.process_input(input=input_text)

        # Generate response based on experience
        relevant_experiences = self._find_relevant_experiences(processed.key_patterns)
        experiences_text = "\n".join([str(e) for e in relevant_experiences])

        response = self.generate_response(
            processed_input=processed.processed_input,
            past_experiences=experiences_text
        )

        # Store experience
        experience = {
            "timestamp": len(self.experiences),
            "input": input_text,
            "processed": processed.processed_input,
            "response": response.response,
            "confidence": response.confidence,
            "patterns": processed.key_patterns
        }
        self.experiences.append(experience)

        # Learn from feedback if available
        if feedback:
            learning = self.evaluate_outcome(
                response=response.response,
                feedback=feedback
            )

            # Update learning
            self._update_learning(
                processed.key_patterns,
                learning.learning_insight,
                learning.strategy_update
            )

            # Reflect on the experience
            reflection = self.reflect(
                experience=str(experience),
                goal=self.learning_goals[0]
            )

            experience["feedback"] = feedback
            experience["learning"] = learning.learning_insight
            experience["reflection"] = reflection.reflection

        return dspy.Prediction(
            agent_name=self.name,
            response=response.response,
            confidence=response.confidence,
            patterns_detected=processed.key_patterns,
            learning_applied=bool(feedback)
        )

    def _find_relevant_experiences(self, patterns, max_experiences=3):
        """Find past experiences with similar patterns."""
        if not patterns:
            return []

        pattern_list = patterns.split(", ") if isinstance(patterns, str) else patterns
        relevant = []

        for exp in self.experiences:
            exp_patterns = exp.get("patterns", "").split(", ") if exp.get("patterns") else []
            overlap = len(set(pattern_list) & set(exp_patterns))
            if overlap > 0:
                relevant.append((exp, overlap))

        # Sort by relevance and return top matches
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [exp[0] for exp in relevant[:max_experiences]]

    def _update_learning(self, patterns, insight, strategy_update):
        """Update learned patterns and strategies."""
        # Update patterns
        for pattern in patterns.split(", "):
            if pattern not in self.patterns:
                self.patterns[pattern] = []
            self.patterns[pattern].append(insight)

        # Update strategies
        if strategy_update:
            key = patterns[:50]  # Use first 50 chars as key
            self.strategies[key] = strategy_update

    def get_learning_summary(self):
        """Get summary of learning progress."""
        return {
            "total_experiences": len(self.experiences),
            "patterns_learned": len(self.patterns),
            "strategies_developed": len(self.strategies),
            "recent_performance": self.performance_history[-5:] if self.performance_history else []
        }
```

## Real-World Agent Applications

### Customer Service Agent

```python
class CustomerServiceAgent(dspy.Module):
    def __init__(self, company_name, knowledge_base):
        super().__init__()
        self.company_name = company_name
        self.knowledge_base = knowledge_base

        self.classify_intent = dspy.Predict("customer_message -> intent, urgency, sentiment")
        self.search_knowledge = dspy.Retrieve(k=3)
        self.generate_response = dspy.ChainOfThought(
            "intent, sentiment, knowledge, company_policy -> response, action_needed"
        )
        self.escalate = dspy.Predict("issue, customer_details -> escalation_reason, department")

        # Session management
        self.sessions = {}

    def forward(self, customer_message, session_id=None):
        # Get or create session
        if not session_id:
            session_id = len(self.sessions)
        session = self.sessions.get(session_id, {"history": [], "context": {}})

        # Classify the customer's intent
        classification = self.classify_intent(customer_message=customer_message)

        # Search knowledge base
        relevant_kb = self.search_knowledge(
            query=f"{classification.intent} {customer_message}"
        )

        # Generate response
        response = self.generate_response(
            intent=classification.intent,
            sentiment=classification.sentiment,
            knowledge="\n".join(relevant_kb.passages),
            company_policy=self.knowledge_base.get("policies", "")
        )

        # Determine if escalation is needed
        action_needed = response.action_needed.lower() if response.action_needed else ""
        if "escalate" in action_needed or "urgent" in classification.urgency.lower():
            escalation = self.escalate(
                issue=customer_message,
                customer_details=str(session["context"])
            )
            final_action = f"Escalated to {escalation.department}: {escalation.escalation_reason}"
        else:
            final_action = response.action_needed

        # Update session
        session["history"].append({
            "timestamp": len(session["history"]),
            "customer_message": customer_message,
            "agent_response": response.response,
            "intent": classification.intent,
            "sentiment": classification.sentiment,
            "action": final_action
        })
        self.sessions[session_id] = session

        return dspy.Prediction(
            session_id=session_id,
            response=response.response,
            intent=classification.intent,
            sentiment=classification.sentiment,
            action_required=final_action,
            agent_name=f"{self.company_name} Support Agent"
        )
```

### Research Assistant Agent

```python
class ResearchAssistantAgent(dspy.Module):
    def __init__(self, research_domains):
        super().__init__()
        self.research_domains = research_domains
        self.plan_research = dspy.ChainOfThought("research_question -> research_plan, methodology")
        self.search_papers = dspy.Retrieve(k=10)
        self.analyze_papers = dspy.Predict("papers, research_question -> key_findings, methodologies")
        self.synthesize_insights = dspy.ChainOfThought(
            "findings, methodologies, research_question -> synthesis, knowledge_gaps"
        )
        self.suggest_next_steps = dspy.Predict("synthesis, gaps -> next_research_steps"

        # Research state
        self.active_research = {}

    def forward(self, research_question, research_id=None):
        if not research_id:
            research_id = len(self.active_research)

        # Plan the research
        planning = self.plan_research(research_question=research_question)

        # Search for relevant papers
        search_results = self.search_papers(query=research_question)

        # Analyze the papers
        analysis = self.analyze_papers(
            papers="\n".join(search_results.passages),
            research_question=research_question
        )

        # Synthesize insights
        synthesis = self.synthesize_insights(
            findings=analysis.key_findings,
            methodologies=analysis.methodologies,
            research_question=research_question
        )

        # Suggest next steps
        next_steps = self.suggest_next_steps(
            synthesis=synthesis.synthesis,
            gaps=synthesis.knowledge_gaps
        )

        # Store research state
        self.active_research[research_id] = {
            "question": research_question,
            "plan": planning.research_plan,
            "papers_found": search_results.passages,
            "findings": analysis.key_findings,
            "synthesis": synthesis.synthesis,
            "next_steps": next_steps.next_research_steps,
            "status": "in_progress"
        }

        return dspy.Prediction(
            research_id=research_id,
            research_plan=planning.research_plan,
            key_findings=analysis.key_findings,
            synthesis=synthesis.synthesis,
            knowledge_gaps=synthesis.knowledge_gaps,
            next_steps=next_steps.next_research_steps,
            papers_analyzed=len(search_results.passages)
        )
```

### Personal Finance Agent

```python
class PersonalFinanceAgent(dspy.Module):
    def __init__(self, user_profile=None):
        super().__init__()
        self.user_profile = user_profile or {}
        self.analyze_transaction = dspy.Predict("transaction -> category, necessity, impact")
        self.assess_financial_health = dspy.ChainOfThought(
            "income, expenses, goals -> health_score, recommendations"
        )
        self.suggest_optimization = dspy.Predict(
            "spending_patterns, financial_goals -> optimization_suggestions"
        )
        self.predict_future = dspy.Predict("current_trends, income_stability -> future_outlook"

        # Financial data
        self.transactions = []
        self.goals = []

    def add_transaction(self, amount, description, date):
        """Add a financial transaction."""
        analysis = self.analyze_transaction(
            transaction=f"{description}: ${amount} on {date}"
        )

        transaction = {
            "id": len(self.transactions),
            "amount": amount,
            "description": description,
            "date": date,
            "category": analysis.category,
            "necessity": analysis.necessity,
            "impact": analysis.impact
        }

        self.transactions.append(transaction)
        return transaction

    def get_financial_advice(self):
        """Get personalized financial advice."""
        # Calculate totals
        income = sum(t["amount"] for t in self.transactions if t["amount"] > 0)
        expenses = sum(abs(t["amount"]) for t in self.transactions if t["amount"] < 0)

        # Assess financial health
        health = self.assess_financial_health(
            income=str(income),
            expenses=str(expenses),
            goals=", ".join(self.goals)
        )

        # Analyze spending patterns
        spending_by_category = {}
        for t in self.transactions:
            if t["amount"] < 0:  # Expense
                cat = t["category"]
                spending_by_category[cat] = spending_by_category.get(cat, 0) + abs(t["amount"])

        # Get optimization suggestions
        optimization = self.suggest_optimization(
            spending_patterns=str(spending_by_category),
            financial_goals=", ".join(self.goals)
        )

        # Predict future outlook
        future = self.predict_future(
            current_trends=str(spending_by_category),
            income_stability="stable"  # Could be more sophisticated
        )

        return dspy.Prediction(
            health_score=health.health_score,
            recommendations=health.recommendations,
            optimization_suggestions=optimization.optimization_suggestions,
            future_outlook=future.future_outlook,
            spending_breakdown=spending_by_category
        )

    def set_goal(self, goal_description, target_amount, deadline):
        """Set a financial goal."""
        goal = {
            "id": len(self.goals),
            "description": goal_description,
            "target": target_amount,
            "deadline": deadline,
            "status": "active"
        }
        self.goals.append(goal)
        return goal
```

## Optimizing Agent Behavior

### Using MIPRO for Agent Decision Making

```python
class OptimizedDecisionAgent(dspy.Module):
    def __init__(self, decision_context):
        super().__init__()
        self.context = decision_context
        self.analyze_situation = dspy.ChainOfThought(
            "situation, context -> situation_analysis, key_factors"
        )
        self.consider_options = dspy.Predict(
            "analysis, factors, constraints -> options, pros_cons"
        )
        self.make_decision = dspy.ChainOfThought(
            "analysis, options, pros_cons, objectives -> decision, confidence, reasoning"
        )

    def forward(self, situation, constraints=None, objectives=None):
        # Analyze the situation
        analysis = self.analyze_situation(
            situation=situation,
            context=self.context
        )

        # Consider available options
        options = self.consider_options(
            analysis=analysis.situation_analysis,
            factors=analysis.key_factors,
            constraints=constraints or "No explicit constraints"
        )

        # Make decision
        decision = self.make_decision(
            analysis=analysis.situation_analysis,
            options=options.options,
            pros_cons=options.pros_cons,
            objectives=objectives or "Optimize outcomes"
        )

        return dspy.Prediction(
            decision=decision.decision,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            alternatives=options.options
        )

# Training data for agent decision making
decision_trainset = [
    dspy.Example(
        situation="Customer reports system downtime affecting 1000 users",
        context="Technical support with SLA requirements",
        constraints="Must resolve within 1 hour, limited team available",
        objectives="Minimize downtime, maintain customer satisfaction",
        decision="Escalate to senior engineers, provide customer updates",
        confidence=0.9
    ),
    # ... more decision scenarios
]

# Optimize decision making
mipro_optimizer = MIPRO(
    metric=decision_quality_metric,
    num_candidates=10
)
optimized_agent = mipro_optimizer.compile(
    OptimizedDecisionAgent("Customer Support System"),
    trainset=decision_trainset
)
```

## Best Practices for Building Agents

### 1. Clear Goal Definition

```python
class GoalOrientedAgent(dspy.Module):
    def __init__(self, primary_goal, sub_goals=None):
        super().__init__()
        self.primary_goal = primary_goal
        self.sub_goals = sub_goals or []
        self.evaluate_alignment = dspy.Predict(
            "action, goal -> alignment_score, alignment_reason"
        )

    def is_goal_aligned(self, action):
        """Check if action aligns with goals."""
        evaluation = self.evaluate_alignment(
            action=str(action),
            goal=self.primary_goal
        )
        return float(evaluation.alignment_score) > 0.7
```

### 2. Robust Error Handling

```python
class ResilientAgent(dspy.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.handle_error = dspy.Predict("error, context -> recovery_action")
        self.fallback_responses = [
            "I'm having trouble processing that. Could you rephrase?",
            "Let me try a different approach.",
            "I need more information to help with that."
        ]

    def safe_execute(self, action_func, *args, **kwargs):
        """Execute action with error handling."""
        try:
            return action_func(*args, **kwargs)
        except Exception as e:
            # Handle error gracefully
            recovery = self.handle_error(
                error=str(e),
                context=str(args)
            )
            return {
                "success": False,
                "error": str(e),
                "recovery_action": recovery.recovery_action
            }
```

### 3. Continuous Learning

```python
class AdaptiveAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.learn_from_feedback = dspy.ChainOfThought(
            "action, outcome, feedback -> learning_insight, strategy_adjustment"
        )
        self.success_patterns = []
        self.failure_patterns = []

    def process_feedback(self, action, outcome, feedback):
        """Learn from feedback on actions."""
        learning = self.learn_from_feedback(
            action=str(action),
            outcome=str(outcome),
            feedback=feedback
        )

        if "success" in feedback.lower():
            self.success_patterns.append(learning.learning_insight)
        else:
            self.failure_patterns.append(learning.learning_insight)

        return learning
```

## Key Takeaways

1. **Intelligent agents combine** perception, planning, and action
2. **Different agent types** suit different use cases and requirements
3. **Memory and learning** are crucial for agent effectiveness
4. **Real-world agents** must handle uncertainty, errors, and feedback
5. **Optimization improves** decision-making and problem-solving
6. **Collaboration enables** agents to tackle complex tasks together

## Next Steps

In the final section of this chapter, we'll explore **Code Generation**, showing how to build automated programming assistants that can help developers write, debug, and optimize code.