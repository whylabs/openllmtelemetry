import inspect
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import wrapt

import openllmtelemetry

F = TypeVar('F', bound=Callable[..., Any])


def trace_task(func: Optional[F] = None,
               *,
               name: Optional[str]=None,
               parameter_names: Optional[List[str]]=None,
               metadata:  Optional[Dict[str, str]]=None
) -> Union[Callable[[F], F], F]:
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        tracer = openllmtelemetry.get_tracer()
        task_name = wrapped.__name__ if name is None else name
        with tracer.start_as_current_span(task_name) as span:
            span.set_attribute("task.decorated.function", str(wrapped.__name__))
            try:
                if metadata:
                    for key, value in metadata.items():
                        span.set_attribute(str(key), str(value))

                if parameter_names:
                    # TODO: consider if there is a better way to get these
                    arg_names = list(inspect.signature(wrapped).parameters.keys())

                    # trace the specified parameters from positional arguments
                    if args:
                        for i, arg_name in enumerate(arg_names):
                            if arg_name in parameter_names and i < len(args):
                                span.set_attribute(arg_name, str(args[i]))

                    # trace the specified parameters from keyword arguments
                    if kwargs:
                        for arg_name, arg_value in kwargs.items():
                            if arg_name in parameter_names:
                                span.set_attribute(arg_name, str(arg_value))
            except Exception:
                pass
            result = wrapped(*args, **kwargs)
        return result

    if func is None:
        return wrapper
    else:
        return wrapper(func)

