- Start Date: 2025-07-17
- RFC PR: https://github.com/thousandbrainsproject/tbp.monty/pull/382

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED",  "MAY", and "OPTIONAL" in this document are to be interpreted as described in [BCP 14](https://www.rfc-editor.org/info/bcp14), [IETF RFC 2119](https://www.rfc-editor.org/rfc/rfc2119), [IETF RFC 8174](https://www.rfc-editor.org/rfc/rfc8174) when, and only when, they appear in all capitals, as shown here.

This guidance does not dictate the only way to implement functionality. There are many ways to implement any particular functionality, each of which will work. This guidance _**establishes constraints**_ so that as more functionality is implemented and functionality is changed, it remains as easy as it was with the first piece of functionality.

Please note that there are differences between research and platform requirements. While research needs speed and agility, the platform needs modularity and stability. These are different and can conflict. The guidance here is for the platform. _**If you are a researcher, you MAY ignore this guidance in your prototype and code in the way most effective for you and your task.**_ Later, if your prototype works and needs to be integrated into Monty, then we will refactor the prototype to correspond to the guidance here.

# Guidance

## Abstract classes MAY be used to specify interfaces without implementations:

_Why?_ We want to move to [Protocols](https://typing.python.org/en/latest/spec/protocol.html#protocols) eventually. Keeping the abstract classes free of implementation makes it easier to transition to Protocols in the future. Whereas, having an abstract class with some implementation requires additional refactoring when we transition to Protocols.

_Why do we want to move to Protocols eventually?_ We want to catch errors as early as possible, and using Protocols allows us to do this at type check time. Using abstract classes delays this until class instantiation, once the program runs.

```python
# ABCs raises errors during instantiation when/if constructor is called:
class Monty(ABC):
	def implemented(self):
		pass

	@abstractmethod
	def unimplemented(self):
		pass

class DefaultMonty(Monty):
	pass

def invoke(monty: Monty):
	monty.unimplemented()

monty = DefaultMonty() # runtime error & fails type check
invoke(monty)  # OK, no type error

# ---

# Typical inheritance raises errors during runtime when monty.unimplemented() is called:
class Monty:
	def implemented(self):
		pass

	def unimplemented(self):
		raise NotImplementedError

class DefaultMonty(Monty):
	pass

def invoke(monty: Monty):
	monty.unimplemented()  # runtime error

monty = DefaultMonty() # OK, no type error
invoke(monty) # OK, no type error

# ---

# Protocols raise errors during type check when attempting use:
class MontyProtocol(Protocol):
	def implemented(self): ...
	def unimplemented(self): ...

class DefaultMonty:
	def implemented(self):
		pass

def invoke(monty: MontyProtocol):
	monty.unimplemented()  # runtime error

monty = DefaultMonty()  # OK
other: MontyProtocol = DefaultMonty() # fails type check
invoke(monty)  # fails type check
```

While abstract classes MAY be used, you SHOULD prefer Protocols.

## Protocols SHOULD be preferred to document usage and expectations

Protocols document a _behaves-like-a_ relationship.

_Why_: We want to catch errors as early as possible, and using Protocols allows us to do this at type check time. Using abstract classes delays this until class instantiation, once the program runs.

There is no material difference **in the context of usage and expectation documentation** between using Protocols and abstract classes. In other contexts, Protocols are favorable because they allow us to raise errors at type check time and, due to structural typing, do not require inheritance.

```python
# Protocols raise errors during type check when monty.unimplemented() is written:
class MontyProtocol(Protocol):
	def implemented(self): ...
	def unimplemented(self): ...

class DefaultMonty:
	def implemented(self):
		pass

monty: MontyProtocol = DefaultMonty() # ok
monty.unimplemented() # fails type check
```

## Inheritance hierarchy SHALL have at most one level of inheritance

_Why_: Inheritance hierarchy allows for overriding methods. As class hierarchies deepen, override analysis becomes more complex. The issue is not how the code functions but the difficulty of reasoning about behavior when multiple layers of overrides are possible. The deeper the hierarchy, the more difficult it is to track what code a specific instance uses, and it makes it unclear where functionality should be overridden. Modifying code with a deep inheritance hierarchy is also complex, in that any change can have cascading effects up and down the hierarchy.

Most of the time, **you should default to not using inheritance hierarchy, and instead, reach for other ways to assemble functionality**. Inheritance is appropriate for an _is-a_ relationship, but this is quite a rare occurrence in practice. A lot of things seem like they form an _is-a_ relationship, but the odds of that relationship being maintained drop off dramatically as the code evolves and the hierarchy deepens.

```python
class Rectangle:
	def __init__(self, length: float, height: float) -> None:
		super().__init__() # See "You SHOULD always include call to super().__init__() ..." section below
		self._length = length
		self._height = height

	@property
	def area(self) -> float:
		return self._length * self._height

class Square(Rectangle):
	def __init__(self, side: float) -> None:
		super().__init__(side, side)

# So far so good...

# The next day, we want to add resize functionality

class Rectangle:

	# --unchanged code omitted-

	def resize(self, new_length: float, new_height: float) -> None:
		self._length = new_length
		self._height = new_height

# But, now this no longer makes sense for the Square
sq = Square(5)
sq.resize(5,3) # ?!
```

As depicted in the example above, if we assume an _is-a_ relationship as the default and reach for inheritance, we can very rapidly introduce functionality that violates the _is-a_ relationship requirement.

Using composition by default instead:

```python
# We want to reuse the area calculating functionality, hence this class
class DefaultAreaComputer:
	@staticmethod
	def area(length: float, height: float) -> float:
		return length * height

class Rectangle:
	def __init__(self, length: float, height: float) -> None:
		super().__init__()
		self._length = length
		self._height = height
		self._area_computer = DefaultAreaComputer

	@property
	def area(self) -> float:
		return self._area_computer.area(self._length, self._height)

class Square:
	def __init__(self, side: float) -> None:
		super().__init__()
		self._side = side
		self._area_computer = DefaultAreaComputer

	@property
	def area(self) -> float:
		return self._area_computer.area(self._side, self._side)

# Now, we want to implement resize for Rectangle

class Rectangle:

	# --unchanged code omitted--

	def resize(self, new_length: float, new_height: float) -> None:
		self._length = new_length
		self._height = new_height

# No issues, because we never assumed is-a relationship in the first place.
```

What if we now want to replace the DefaultAreaComputer with a different implementation?

```python
# Implement a different computer
class TooComplicatedAreaComputer:
	@staticmethod
	def area(length: float, height: float) -> float:
		return 4 * (length / 2) * (height / 2)

# Use new computer in Rectangle
class Rectangle:
	def __init__(self, length: float, height: float) -> None:
		super().__init__()
		self._length = length
		self._height = height
		self._area_computer = TooComplicatedAreaComputer

	# --unchanged code omitted--
```

What if we want to make the area computer configurable?

```python
# Define the protocol
class AreaComputer(Protocol):
	@staticmethod
	def area(length: float, height: float) -> float: ...

# Update Rectangle to accept area computer
class Rectangle:
	def __init__(
		self,
		length: float,
		height: float,
		area_computer: type[AreaComputer] = TooComplicatedAreaComputer
	) -> None:
		super().__init__()
		self._length = length
		self._height = height
		self._area_computer = area_computer

	# --unchanged code omitted--
```

If we want our code to change rapidly, to try out different ideas, and to configure existing code with these variants, using modular components for functionality reuse instead of inheritance allows for changes to remain small in scope without affecting unrelated functionality up and down the inheritance chain.

## Bare functions or static methods SHOULD be used to share a functionality implementation that does not access instance state

_Why_: Do not require state that you don’t access. Functions without state are vastly easier to reuse, refactor, reason about, and test.

```python
# calculating an area
class DefaultAreaComputer:
	@staticmethod
	def area(length: float, height: float) -> float:
		return length * height

# alternatively
def area(length: float, height: float) -> float:
	return length * height
```

A reason to use a static method on a class over a bare function would be when we want to pass the functionality to another class. This is because we want our configurations to be serializable, and a type is serializable in a more straightforward manner than a Callable would be. For example:

```python
# Static method approach
class Rectangle:
	def __init__(
		self,
		length: float,
		height: float,
		area_computer: type[AreaComputer] = DefaultAreaComputer # Easier to serialize
	) -> None:
		super().__init__()
		self._length = length
		self._height = height
		self._area_computer = area_computer

	@property
	def area(self) -> float:
		return self._area_computer.area(self._length, self._height)

# Bare function approach
class Rectangle:
	def __init__(
		self,
		length: float,
		height: float,
		area_computer: Callable[[float, float], float] = area # More challenging to serialize
	) -> None:
		super().__init__()
		self._length = length
		self._height = height
		self._area_computer = area_computer

	@property
	def area(self) -> float:
		return self._area_computer(self._length, self._height)
```

## To share a functionality implementation that reads the state of the instance being mixed with, mixins MAY be used

For sharing functionality, mixins only implement a shared behaves-like-a functionality. They add functionality, however, mixins SHALL NOT add state to the instance being mixed with. Every time you find yourself in need of state when working on a mixin, switch to composition instead.

_Why_: When Mixins do not add state, they are not terrible for implementing shared functionality. That’s why you MAY use them for this. However, when Mixins add state, you must look at two places for the state to understand the implementation instead of one. Having to look in two places is an example of incidental complexity, where it is not inherent to the problem being solved. Incidental complexity should be minimized.

```python
# OK, Mixin only reads state
class RectangleAreaMixin:
	@property
	def area(self) -> float:
		return self._length * self._height

class Rectangle(RectangleAreaMixin):
	def __init__(self, length: float, height: float) -> None:
		super().__init__()
		self._length = length
		self._height = height

# ---

# Not OK, Mixin adds state
class RectangleAreaMixin:
	def __init__(self, length: float, height: float) -> None:
		super().__init__()
		self._length = length
		self._height = height

	@property
	def area(self) -> float:
		return self._length * self.height

class Rectangle(RectangleAreaMixin):
	def __init__(self, length: float, height: float) -> None:
		super().__init__(length, height)
```

## Composition SHOULD be used to share a functionality implementation that needs its own state

Composition is used to implement a _has-a_ relationship.

_Why_: Components encapsulate additional state in a single concept in a single place in the code.

Given a Rectangle that uses a DefaultAreaComputer (because we reuse that functionality elsewhere), let’s say we want to count how many times we resized it.

```python
class DefaultAreaComputer:
	@staticmethod
	def area(length: float, height: float) -> float:
		return length * height

class Rectangle:
	def __init__(self, length: float, height: float) -> None:
		super().__init__()
		self._length = length
		self._height = height
		self._area_computer = DefaultAreaComputer
		self._resize_count = 0 # We track count in Rectangle state

	@property
	def area(self) -> float:
		return self._area_computer.area(self._length, self._height)

	def resize(self, new_length: float, new_height: float) -> None:
		self._length = new_length
		self._height = new_height
		self._resize_count += 1 # We update internal state

	@property
	def resize_count(self) -> int:
		return self._resize_count

# Now, we want to reuse the count functionality

# First, we extract/encapsulate the DefaultCounter functionality
class DefaultCounter:
	def __init__(self) -> None:
		self._count = 0

	def increment(self) -> None:
		self._count += 1

	@property
	def count(self) -> int:
		return self._count

# We then update Rectangle to use the shared functionality
class Rectangle:
	def __init__(self, length: float, height: float) -> None:
		super().__init__()
		self._length = length
		self._height = height
		self._area_computer = DefaultAreaComputer
		# Note that the count itself (state) is no longer in the Rectangle
		self._resize_counter = DefaultCounter() # We track count in DefaultCounter

	# --unchanged code omitted--

	def resize(self, new_length: float, new_height: float) -> None:
		self._length = new_length
		self._height = new_height
		self._resize_counter.increment() # We update the count

	@property
	def resize_count(self) -> int:
		return self._resize_counter.count

# And now that we extracted the DefaultCounter functionality, we can use it elsewhere
class Circle:
	def __init__(self, radius: float) -> None:
		super().__init__()
		self._radius = radius
		# Note that DefaultCounter() introduces new state, but it is
		# encapsulated within the component
		self._resize_counter = DefaultCounter()

	@property
	def area(self) -> float:
		# We don't need to make everything a component.
		# Since we don't reuse circle area functionality anywhere,
		# it is OK to have it inline here.
		return math.pi * self._radius ** 2

	def resize(self, new_radius: float) -> None:
		self._radius = radius
		self._resize_counter.increment()

	@property
	def resize_count(self) -> int:
		return self._resize_counter.count
```

## You SHOULD always include call to super().__init__() in your __init__ methods

_Why_: This avoids possible issues with multiple inheritance by opting into “cooperative multiple inheritance.”

This should not be an issue once all of our code follows this guidance document, specifically ensuring that Mixins do not introduce state, making Mixins with `__init__` unlikely. However, it may be a while before we get there, so this guidance is included.

See https://eugeneyan.com/writing/uncommon-python/#using-super-in-base-classes for additional details, but here are some examples with their corresponding output:

> [!NOTE]
>
> `print` call occurs _after_ the call to `super().__init__()`. Output would be in different order if `print` ocurred before the  `super().__init__()` call.

```python
# Correct and expected

class Parent:
	def __init__(self) -> None:
		super().__init__()
		print("Parent init")

class Mixin:
	pass

class Child(Mixin, Parent):
	def __init__(self) -> None:
		super().__init__()
		print("Child init")

child = Child()

# Output
# > Parent init
# > Child init

# Also correct and expected
class Parent:
	def __init__(self) -> None:
		super().__init__()
		print("Parent init")

class Mixin:
	pass

class Child(Parent, Mixin):
	def __init__(self) -> None:
		super().__init__()
		print("Child init")

child = Child()

# Output
# > Parent init
# > Child init
```

The problems begin when inherited classes all have `__init__` defined.

```python
# Correct and expected

class Parent:
	def __init__(self) -> None:
		super().__init__()
		print("Parent init")

class Mixin:
	def __init__(self) -> None:
		super().__init__()
		print("Mixin init")

class Child(Mixin, Parent):
	def __init__(self) -> None:
		super().__init__()
		print("Child init")

child = Child()

# Output
# > Parent init
# > Mixin init
# > Child init

# Also correct and expected
class Parent:
	def __init__(self) -> None:
		super().__init__()
		print("Parent init")

class Mixin:
	def __init__(self) -> None:
		super().__init__()
		print("Mixin init")

class Child(Parent, Mixin):
	def __init__(self) -> None:
		super().__init__()
		print("Child init")

child = Child()

# Output
# > Mixin init
# > Parent init
# > Child init

# If you skip super().__init__() call in one of the inherited classes, some class __init__ methods are skipped

# class Child(Mixin, Parent) where we skip super().__init__() in Mixin

class Parent:
	def __init__(self) -> None:
		super().__init__()
		print("Parent init")

class Mixin:
	def __init__(self) -> None:
		# super().__init__() skipped
		print("Mixin init")

class Child(Mixin, Parent):
	def __init__(self) -> None:
		super().__init__()
		print("Child init")

child = Child()

# Output
# > Mixin init
# > Child init

# class Child(Parent, Mixin) where we skip super().__init__() in Parent

class Parent:
	def __init__(self) -> None:
		# super().__init__() skipped
		print("Parent init")

class Mixin:
	def __init__(self) -> None:
		super().__init__()
		print("Mixin init")

class Child(Parent, Mixin):
	def __init__(self) -> None:
		super().__init__()
		print("Child init")

child = Child()

# Output
# > Parent init
# > Child init
```
