# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from enum import Enum
from typing import List, Optional, Callable, Any

from loguru import logger


class TaskStatus(Enum):
    """Enum representing the status of a task."""

    PENDING = "pending"  # Task has not started yet
    RUNNING = "running"  # Task is currently running
    COMPLETED = "completed"  # Task has completed successfully
    FAILED = "failed"  # Task has failed
    CANCELLED = "cancelled"  # Task was cancelled


class Task:
    """Base class for tasks that can be executed by the TaskQueue.
    A task represents a unit of work that can be executed asynchronously.
    Subclasses should override the update() method to implement the task's behavior.
    """

    def __init__(self, name: str = None, priority: int = 0):
        """Initialize a new task.
        Args:
            name: Optional name for the task. If not provided, the class name will be used.
            priority: Priority of the task. Higher values indicate higher priority.
        """
        self.name = name or self.__class__.__name__
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.dependencies: List[Task] = []
        self._on_complete_callbacks: List[Callable[[Task], Any]] = []
        self._on_fail_callbacks: List[Callable[[Task], Any]] = []

    def add_dependency(self, task: "Task") -> "Task":
        """Add a dependency to this task.
        The task will not start until all dependencies have completed.
        Args:
            task: The task that must complete before this task can start.
        Returns:
            self: For method chaining.
        """
        self.dependencies.append(task)
        return self

    def on_complete(self, callback: Callable[["Task"], Any]) -> "Task":
        """Register a callback to be called when the task completes successfully.
        Args:
            callback: Function to call when the task completes.
        Returns:
            self: For method chaining.
        """
        self._on_complete_callbacks.append(callback)
        return self

    def on_fail(self, callback: Callable[["Task"], Any]) -> "Task":
        """Register a callback to be called when the task fails.
        Args:
            callback: Function to call when the task fails.
        Returns:
            self: For method chaining.
        """
        self._on_fail_callbacks.append(callback)
        return self

    def update(self) -> Optional[List["Task"]]:
        """Update the task's state.
        This method is called by the TaskQueue on each update cycle.
        Subclasses should override this method to implement the task's behavior.
        Returns:
            Optional[List[Task]]: A list of new tasks to add to the queue, or None.
        """
        return None

    def is_finished(self) -> bool:
        """Check if the task has finished.
        Returns:
            bool: True if the task has finished (completed, failed, or cancelled), False otherwise.
        """
        return self.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ]

    def is_ready(self) -> bool:
        """Check if the task is ready to run.
        A task is ready to run if it is pending and all its dependencies have completed.
        Returns:
            bool: True if the task is ready to run, False otherwise.
        """
        if self.status != TaskStatus.PENDING:
            return False

        return all(dep.status == TaskStatus.COMPLETED for dep in self.dependencies)

    def complete(self) -> None:
        """Mark the task as completed and trigger completion callbacks."""
        self.status = TaskStatus.COMPLETED
        for callback in self._on_complete_callbacks:
            callback(self)

    def fail(self) -> None:
        """Mark the task as failed and trigger failure callbacks."""
        self.status = TaskStatus.FAILED
        for callback in self._on_fail_callbacks:
            callback(self)

    def cancel(self) -> None:
        """Mark the task as cancelled."""
        self.status = TaskStatus.CANCELLED


class TaskQueue:
    """A queue for executing tasks asynchronously.
    The TaskQueue manages a collection of tasks and executes them according to their
    dependencies and priorities.
    """

    def __init__(self):
        """Initialize a new task queue."""
        self.tasks: List[Task] = []
        self.paused = False

    def add_task(self, task: Task) -> Task:
        """Add a task to the queue.
        Args:
            task: The task to add.
        Returns:
            Task: The added task, for method chaining.
        """
        self.tasks.append(task)
        return task

    def add_tasks(self, tasks: List[Task]) -> None:
        """Add multiple tasks to the queue.
        Args:
            tasks: The tasks to add.
        """
        self.tasks.extend(tasks)

    def remove_task(self, task: Task) -> None:
        """Remove a task from the queue.
        Args:
            task: The task to remove.
        """
        if task in self.tasks:
            self.tasks.remove(task)

    def clear(self) -> None:
        """Remove all tasks from the queue."""
        self.tasks.clear()

    def pause(self) -> None:
        """Pause the task queue."""
        self.paused = True

    def resume(self) -> None:
        """Resume the task queue."""
        self.paused = False

    def update(self) -> None:
        """Update all tasks in the queue.
        This method should be called regularly to progress the tasks.
        """
        if self.paused:
            return

        # Sort tasks by priority (higher first)
        self.tasks.sort(key=lambda t: -t.priority)

        # Make a copy of the task list to avoid modification during iteration
        tasks_to_process = list(self.tasks)
        new_tasks = []

        for task in tasks_to_process:
            # Skip tasks that are not ready to run
            if task.status == TaskStatus.PENDING and not task.is_ready():
                continue

            # Start running tasks that are ready
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.RUNNING

            # Update running tasks
            if task.status == TaskStatus.RUNNING:
                try:
                    result = task.update()
                    if result is not None:
                        new_tasks.extend(result)
                except Exception as e:
                    logger.error(f"Error updating task {task.name}: {e}")
                    task.fail()

            # Remove finished tasks
            if task.is_finished():
                self.tasks.remove(task)

        # Add new tasks
        self.add_tasks(new_tasks)

    def is_finished(self) -> bool:
        """Check if all tasks have finished.
        Returns:
            bool: True if there are no tasks in the queue, False otherwise.
        """
        return len(self.tasks) == 0

    def get_task_count(self) -> int:
        """Get the number of tasks in the queue.
        Returns:
            int: The number of tasks.
        """
        return len(self.tasks)

    def get_running_tasks(self) -> List[Task]:
        """Get all running tasks.
        Returns:
            List[Task]: A list of running tasks.
        """
        return [task for task in self.tasks if task.status == TaskStatus.RUNNING]

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks.
        Returns:
            List[Task]: A list of pending tasks.
        """
        return [task for task in self.tasks if task.status == TaskStatus.PENDING]
