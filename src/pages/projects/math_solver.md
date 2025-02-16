---
layout: ../../layouts/ProjectLayout.astro
title: 'Math Solver'
cover: /images/projects/math_solver/free_conversion.png
description: 'Simple web application to solve mathematical problems like unit conversion, symbolic solving, and plotting.'
---
<style>
	.caption {
		text-align: center;
        margin-top: -20px;
	}
</style>

## Visit the project

- [Math Solver](https://math-solver.alex-quiroga.com/)

## Description

The project consist on a web application that allows users to solve mathematical problems like unit conversion, symbolic solving, and plotting. The application uses Python and Flask to create a web server that sends dynamic requests to the user's browser. The HTML templates are created using Jinja2 and [TailwindCSS](https://tailwindcss.com/) to create a responsive and user-friendly interface. More specifically, the front-end uses some some [Flowbite](https://flowbite.com/docs/getting-started/introduction/) components (Navbar and forms) to create a visually appealing interface. Additionally, a interchangeable theme is also implemented using [TailwindCSS](https://tailwindcss.com/), [Dark Mode](https://tailwindcss.com/docs/dark-mode) properties and some JavaScript. 

In order to avoid excessive syntax repetition a `templates/layout.html` is defined with the base HTML including the navbar and the dark/light theme button. The different pages defined in the `templates` folder are the home page, the unit conversion page, the symbolic solving page, the plotting page and the apology page. All the pages use the same layout and share the same navbar.

The business logic is implemented using Python classes that handle the units conversion and symbolic solving. This Python modules are in the `app/models` folder:

- The `conversor.py` module contains classes to handle things like Dimensions, Units, Measurements and Unit systems. A Unit is defined by its name, symbol, list of dimension factors and a callable function to convert it to the [International System of Units](https://en.wikipedia.org/wiki/International_System_of_Units). On the other hand, the measures are defined by its value and the list of unit factors corresponding to that value. There are other objects like `PhysicalDimensions` that defines the dimension factors corresponding to each magnitudes so they can be used  by `AllUnits` class in order to define all the available units.  Then the class `UnitSystems` defines the units corresponding to each unit system and finally the `UnitConversor` class implement the parser to identify unit factors from string and the conversor itself. The code is implemented so more units and systems can be add without extra modifications. In addition, the conversor can also handle currency conversion using the [HexaRate API](https://hexarate.paikama.co/).
- The `symbolic.py` module contains a single class with one method for each symbolic solver. Ir order to parse and obtain the solution the [SymPy](https://www.sympy.org/en/index.html) library is used. First, the `solve_equation` method return the solution from any equation expressed in terms of a single unknown. User has to introduce the expression that is equal to 0 and the symbol corresponding to the unknown. The `solve_integral` and `solve_derivative` do something similar but with the expression for the function to integrate or derivate and passing the integration or derivation variable. Finally, to provide a linear algebra solver the `find_eigenvalues` method is defined, which can find the eigenvalues and eigenvector form a particular matrix introduced like nested lists. This four implemented methods are a small subset of what sympy can do programatically.
- The `plotter.py` module defines a single class which constructor accepts a function expression and their limits. Then the `get_fig()` method convert the function expression to a proper function using sympy and return a [plotly](https://plotly.com/) graph object. This graph is finally converted to JSON and pass it to the template so it can be rendered later in the browser. To do so, a JavaScript variable receive the JSON from a placeholder and runs the plotly API to render the responsive graph.

Multiple forms allow the user to enter input values for different cases. Requests are handled by the endpoints defined in `main/routes.py` which return an apology page in case something goes wrong. For this, the apology html template is rendered. The template receives a message and the error code and renders an image of the [HTTP CATS API](https://http.cat/) along with the specific message if applicable.


## Examples
### Unit conversor

- **Free Units:** Convert units freely defining the units of the input value and the output system.
    - Example: Convert 10 meters (m) to feet (ft).
- **By magnitude:** Convert between different units for a particular magnitude.
    - Example: Convert temperature from Celsius (°C) to Fahrenheit (°F).

### Symbolic solver
- **Equation:** Solve equations analitically from a symbolic expression.
    - Example: Solve the equation `x**2 + 2*x + 1 = 0`.
- **Eigenvalues:** Find the eigenvalues and eigenvectors of a matrix.
    - Example: Find the eigenvalues and eigenvectors of the matrix `[[1, 2], [3, 4]]`.

- **Integral:** Solve the undetermined integral of a function from a symbolic expression.
    - Example: Solve the integral `∫(x**2 + 2*x + 1) dx`.
- **Derivative:** Find the derivative of a function from a symbolic expression.
    - Example: Find the derivative of `sin(x)`.
### Plotter
- **Plot:** Plot a function from its symbolic expression between two limits.
    - Example plot: `tanh(x)` between `-5` and `5`.