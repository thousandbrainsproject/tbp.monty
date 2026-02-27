- Start Date: 2025-07-31
- RFC PR: https://github.com/thousandbrainsproject/tbp.monty/pull/409

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [BCP 14](https://www.rfc-editor.org/info/bcp14) [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) [RFC 8174](https://www.rfc-editor.org/rfc/rfc8174) when, and only when, they appear in all capitals, as shown here.

# Project Typing Guidelines

Python initially introduced type hinting in version 3.5 after being defined in [PEP 484](https://peps.python.org/pep-0484/). The Python interpreter itself generally ignores type hints, so the main way in which they are used is via IDE/LSP support, and through type checking tools like `mypy`. The thinking was for this system to be optional. Any code that would run before would continue to run after the introduction of type hints. It is also in some ways gradual, in that the entire codebase doesn't have to be fully type hinted before some degree of benefit can be realized.

In the future, we would like to utilize type hints in the Monty codebase. However, there are multiple approaches that can be taken to do this, so we want to provide some guidance to ensure we're adding type hints that are giving us the most benefits.

## What does type hinting do for us?

Python already has a _dynamic type_ system. Variables don't have types which restrict the values that can be assigned to them. The values themselves have types that determine what can be done with them, however any check of those types occur at runtime throwing errors when those checks fail.

Using type hints with a type checker implements a _static type_ system. A static type system assigns types to variables, restricting which values can be assigned to them and which operations can be used on those variables and checking those types ahead of time without running the code.

The main benefit of a static type checker enforcing constraints is that it can prevent errors by checking ahead of time for method calls or operations on a variable that would otherwise throw an error at runtime. Type hints also document what arguments are allowed in methods and what types they return. They can also encode logic into the type system to allow for proving certain properties of the code.

## Type hints do nothing without a type checker

One thing to be aware of, the Python interpreter DOES NOT care about type hints. Anything you could normally do in Python without type hints will still be possible at runtime with type hints. The only value they add is when a type checker is used to confirm that the operations being performed match what the type hints indicate is allowed.

This especially applies to newtypes (see below for details). Newtypes do nothing at runtime. There’s a _slight_ performance hit for the call to the “constructor” of the newtype, but it’s negligible. It is, like the rest of type hinting, just a hint, that the Python interpreter ignores, but a type checker can use to ensure correctness.

## Guidelines

### Methods and functions SHOULD accept the broadest possible type

The type assigned to arguments should be as abstract as possible when specifying the arguments to a method or function. For example, if a method needs some collection of items, instead of specifying the type as a `List`, use `Iterable` instead.

```python
# Don't restrict the argument to a List
def double_list(l: List[int]) -> List[int]:
    return [x * 2 for x in l]

# Instead, use an appropriate collection type
def double_coll(c: Iterable[int]) -> List[int]:
    return [x * 2 for x in c]
```

Using the broadest type that provides the functionality needed allows for more flexibility when calling the function or method. The second example can be called with a set, while the first only works for lists.

### Methods and functions SHOULD return the narrowest possible type

The type returned from a method or function should be the most concrete type possible. Similar to the previous example, the return type could be `Collection[int]`, but that type wouldn't allow for calling list-specific methods on the returned value, even though we're returning a list.

Returning the most concrete type gives more flexibility to the caller to use that value in ways that the function/method author might not have considered.

### Structural typing SHOULD NOT be used when possible.

_Structural typing_ is a type system in which the structure of the type is what matters for type checking. Two values with the same structure will type-check as the same type, regardless of any type aliases being used (see the Python glossary for a stricter definition of [structural types](https://typing.python.org/en/latest/spec/glossary.html#term-structural)).

Basic types like `str`, `list`, and `dict[int, str]` are examples of structural types in Python. Nothing about a `dict[int, str]` indicates what those keys or values represent, and any dictionary that takes integers as keys and has strings as values would type-check against that type.

Type aliases are structural types; they **do not** lead to nominal types. From the type checker's perspective, they are replaced with the type that they alias. They are only a convenience for not having to write out long structural types.

While Protocols provide structural subtyping, they are an exception to this guidance and SHOULD be used for _static duck typing_ (see [the section on duck typing](#protocols-should-be-used-to-define-behaviour-that-would-normally-be-duck-typed) for details).

Structural types SHOULD NOT be used when possible because they don't define the concepts that the types represent, reducing their usefulness.

```python
# Type alias for a quaternion
# This doesn't define a new type, it just allows the function
# definition below to be shorter.
QuaternionWXYZ = Tuple[float, float, float, float]
# Define another one for a different order
QuaternionXYZW = Tuple[float, float, float, float]

def normalize_quaternion(quat: QuaternionWXYZ):
    # Based on the type alias, the function expects this
    w, x, y, z = quat
    return normalized_quaterion

quat: QuaternionXYZW = (0.0, 0.0, 0.0, 1.0) 
norm = normalize_quaternion(quat) # This type checks, but gives invalid results
```

Alternative ways to model a quaternion are using newtypes or dataclasses, both forms of _nominal types_ (see [the section on nominal typing](#nominal-typing-should-be-used-to-name-concepts-using-types) for details on the benefits of nominal types). Which to choose would depend on whether additional functionality is needed, or for easier compatibility with third-party libraries.

```python
@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float

# Or
QuaternionWXYZ = NewType("QuaternionWXYZ", Tuple[float, float, float, float])
```

The dataclass approach forces the author to specify which coefficient they want to access, removing ambiguity, while the newtype still requires the author to be careful about which `float` in the `tuple` is the one they want, but the name helps indicate which it is. The newtype also prevents passing a raw tuple where a `QuaternionWXYZ` is expected without explicitly turning it into one.

An exception to this guideline would be the types of **internal** fields of a class, like the `float`s in the Quaternion dataclass in the previous example. Oftentimes those don’t need to be anything more specific than the underlying structural type. An example might be a name field on a class, which could simply be a `str`. Unless there is some extra metadata, e.g. the `str` is untrusted user input that should be handled carefully, a simple `str` will suffice. This can also apply to things like lists, sets, and dictionaries.

A rule of thumb in most cases would be to ask whether the type needs to be exposed outside of the class or function in which the variable is defined. Function or method arguments and return types should generally avoid structural types in favor of nominal types.

### Nominal typing SHOULD be used to name concepts using types

_Nominal typing_ is a type system in which the name of the type is what matters for type checking. Two values with the same structure but different names are distinct types in this system (see the Python glossary for a stricter definition of [nominal type](https://typing.python.org/en/latest/spec/glossary.html#term-nominal)).

More complex types, like dataclasses and regular classes, are examples of nominal types in Python. Two dataclasses with the same fields but different names would type-check as different types. Classes and dataclasses (which are a convenience for defining certain kinds of classes) are more opaque than anything to do with the fields or methods they have. Different classes are different types and the only way instances of those classes can type check as each other is if one is a subclass of another (this is the [_subtype polymorphism_](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)#Subtyping) behaviour provided by object-oriented languages).

Python also provides `NewType`s which can be used to define nominal types for basic types that would normally be structurally typed. An example of a newtype would be to define the concept of radians and degrees for angles in a system, and ensure that they can't be used in the wrong places.

```python
Radians = NewType("Radians", float)
Degrees = NewType("Degrees", float)

# Both Radians and Degrees are floats, but they cannot be swapped.
def rads_to_degrees(rads: Radians) -> Degrees:
    pass

right_angle = Radians(Math.pi / 4)
another_angle = Degrees(180.0)

rads_to_degrees(right_angle) # works fine
# FAILS to typecheck: "Degrees" is not "Radians", etc.
rads_to_degrees(another_angle)
# FAILS to typecheck: "float" is not "Radians", etc.
rads_to_degrees(270.0)
```

Another example would be to codify the order of a quaternion (since there is disagreement between libraries whether to use WXYZ or XYZW) and then check usages (see the quaternion example in [the section on structural typing](#structural-typing-should-not-be-used-when-possible)).

Newtypes should be used when **no additional functionality** beyond the underlying type is needed, but metadata about the type needs to be tracked. An example of this would be unsafe and safe strings in a web application. They shouldn’t be confused because they can lead to security vulnerabilities, but with newtypes we can help the author to think about which is being used.

```python
UnsafeString = NewType("UnsafeString", str)
SafeString = NewType("SafeString", str)

def render_with_content(template: SafeString, content: SafeString):
    # This isn't the best way to do this, but it's an example
    return format(template, content)

def sanitize_string(s: UnsafeString) -> SafeString:
    # do some sanitization
    return SafeString(new_s)

unsafe_input: UnsafeString # comes from some user input
template = SafeString("Hello, {}")
# FAILS to typecheck: "UnsafeString" is not "SafeString"
render_with_content(template, unsafe_input)

safe_input = sanitize_string(unsafe_input)
# FAILS to typecheck: "SafeString" is not "UnsafeString"
safe_input = sanitize_string(safe_input)
```

In this example, it becomes more difficult to accidentally pass an unsafe string to a rendering function that could cause security issues, and it’s also difficult to accidentally sanitize a safe string a second time.

### Protocols SHOULD be used to define behaviour that would normally be duck typed

_Duck typing_ is a form of structural typing in dynamically typed languages where two different value types can be substituted for each other if they have the same methods. The name comes from the duck test. “If it walks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck.”

Python’s type hinting library provides Protocols to allow defining the shared interface that types can implement in order to be considered the same type by a type checker. They allow defining interfaces that other types use without requiring those types to inherit from a base class to do it. A Protocol can be implemented by types elsewhere without those types even knowing about the Protocol’s existence.

Protocols allow for defining abstract interfaces that types can satisfy to allow functions and methods to accept the broadest possible types. The abstract collection types, e.g. `Collection`, `Iterable`, etc., in the standard library can be thought of like protocols that various concrete collection types implement. This can be seen in the first example above.

### The `Any` type SHOULD NOT be used

The `Any` type in Python causes the type checker to stop attempting to do type checking, since it has no information whatsoever to go on. This means we lose all the benefits of using a static type checker, and thus it SHOULD NOT be used.

Another similar type is `object`. While it can be used with arbitrary typed values similar to `Any`, it is an ordinary static type so the type checker will reject most operations on it (i.e., those not defined for all objects), and so it MAY be used where appropriate (see the Mypy documentation on [Any vs. object](https://mypy.readthedocs.io/en/stable/dynamic_typing.html#any-vs-object) for more details).

### Third-party libraries with poor type hinting SHOULD be isolated as much as possible.

Some third-party libraries, especially ones that are extensions written in another language like C, provide poor type hinting, meaning the majority of the benefits that come from static type checking aren't available. To minimize this surface area, we should isolate code that uses these third-party libraries, wrapping them in functions or methods that provide the correct types that the rest of our code expects.

Even in libraries like NumPy that purport to have typing hints available, sometimes those are less useful. For example, a lot of functions return `npt.NDArray[Any]` when we know for certain that the type should be `npt.NDArray[np.float64]`. In cases like these, we want to isolate the chaos and return the types we want.

```python
def do_maths(input: ...) -> npt.NDArray[np.float64]:
    # do lots of Numpy calls
    return result
```

This might require the use of [`typing.cast()`](https://docs.python.org/3/library/typing.html#typing.cast) or explicit type hints on variable declarations to satisfy the type checker.

The granularity SHOULD NOT be individual NumPy functions, but rather whole operations in our code where we are using multiple NumPy functions in a row. We're not trying to make a wrapper for poorly typed libraries, but making sure when we return things into our code, store values on objects, etc. that we're giving them useful types.

```python
# Don't do something this granular
def mujoco_worldbody(spec: MjSpec) -> MjsBody:
    return spec.worldbody  # this might require `cast(MjsBody, spec.worldbody)`

def some_other_method(...) -> None:
    ...
    worldbody = mujoco_worldbody(spec)
    ...

# Instead, just declare the types in the other method where needed
def some_other_method(...) -> MjsBody:
   ...
   worldbody: MjsBody = spec.worldbody
   # More MuJoCo calls on worldbody that may need type hints
   ...
```

This would also be a good place to use newtypes to define _input_ argument types like quaternion tuples that have a particular order so that callers don't pass the wrong values into these external library functions. See [the nominal type guidance](#nominal-typing-should-be-used-to-name-concepts-using-types) above.
