from typing import Callable, List, Optional

ContentIdProvider = Callable[[List[str]], Optional[str]]
