import ast

class CodeComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.loop_count = 0
        self.recursive_calls = 0
        self.function_calls = 0
        self.if_conditions = 0
    
    def visit_For(self, node):
        """Detect for-loops"""
        self.loop_count += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        """Detect while-loops"""
        self.loop_count += 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Detect recursion in function definitions"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and child.func.id == node.name:
                self.recursive_calls += 1
        self.generic_visit(node)

    def visit_If(self, node):
        """Detect if-conditions"""
        self.if_conditions += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        """Detect function calls"""
        self.function_calls += 1
        self.generic_visit(node)

def analyze_code(code):
    """Runs AST parser on the given Python code"""
    tree = ast.parse(code)
    analyzer = CodeComplexityAnalyzer()
    analyzer.visit(tree)

    return {
        "Loops": analyzer.loop_count,
        "Recursion": analyzer.recursive_calls,
        "Function Calls": analyzer.function_calls,
        "If Conditions": analyzer.if_conditions
    }

# Example Code Snippet
code_snippet = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

for i in range(10):
    print(factorial(i))
"""

# Run Analysis
features = analyze_code(code_snippet)
print("Extracted Features:", features)
