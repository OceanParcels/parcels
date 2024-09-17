import ast
import collections
import math
import random
import warnings
from abc import ABC
from copy import copy

import cgen as c

from parcels.field import Field, NestedField, VectorField
from parcels.grid import Grid
from parcels.particle import JITParticle
from parcels.tools.statuscodes import StatusCode
from parcels.tools.warnings import KernelWarning


class IntrinsicNode(ast.AST):
    def __init__(self, obj, ccode):
        self.obj = obj
        self.ccode = ccode


class FieldSetNode(IntrinsicNode):
    def __getattr__(self, attr):
        if isinstance(getattr(self.obj, attr), Field):
            return FieldNode(getattr(self.obj, attr), ccode=f"{self.ccode}->{attr}")
        elif isinstance(getattr(self.obj, attr), NestedField):
            if isinstance(getattr(self.obj, attr)[0], VectorField):
                return NestedVectorFieldNode(getattr(self.obj, attr), ccode=f"{self.ccode}->{attr}")
            else:
                return NestedFieldNode(getattr(self.obj, attr), ccode=f"{self.ccode}->{attr}")
        elif isinstance(getattr(self.obj, attr), VectorField):
            return VectorFieldNode(getattr(self.obj, attr), ccode=f"{self.ccode}->{attr}")
        else:
            return ConstNode(getattr(self.obj, attr), ccode=f"{attr}")


class FieldNode(IntrinsicNode):
    def __getattr__(self, attr):
        if isinstance(getattr(self.obj, attr), Grid):
            return GridNode(getattr(self.obj, attr), ccode=f"{self.ccode}->{attr}")
        elif attr == "eval":
            return FieldEvalCallNode(self)
        else:
            raise NotImplementedError("Access to Field attributes are not (yet) implemented in JIT mode")


class FieldEvalCallNode(IntrinsicNode):
    def __init__(self, field):
        self.field = field
        self.obj = field.obj
        self.ccode = ""


class FieldEvalNode(IntrinsicNode):
    def __init__(self, field, args, var, convert=True):
        self.field = field
        self.args = args
        self.var = var  # the variable in which the interpolated field is written
        self.convert = convert  # whether to convert the result (like field.applyConversion)


class VectorFieldNode(IntrinsicNode):
    def __getattr__(self, attr):
        if attr == "eval":
            return VectorFieldEvalCallNode(self)
        else:
            raise NotImplementedError("Access to VectorField attributes are not (yet) implemented in JIT mode")

    def __getitem__(self, attr):
        return VectorFieldEvalNode(self.obj, attr)


class VectorFieldEvalCallNode(IntrinsicNode):
    def __init__(self, field):
        self.field = field
        self.obj = field.obj
        self.ccode = ""


class VectorFieldEvalNode(IntrinsicNode):
    def __init__(self, field, args, var, var2, var3, var4, convert=True):
        self.field = field
        self.args = args
        self.var = var  # the variable in which the interpolated field is written
        self.var2 = var2  # second variable for UV interpolation
        self.var3 = var3  # third variable for UVW interpolation
        self.var4 = var4  # extra variable for sigma-scaling for croco
        self.convert = convert  # whether to convert the result (like field.applyConversion)


class NestedFieldNode(IntrinsicNode):
    def __getitem__(self, attr):
        return NestedFieldEvalNode(self.obj, attr)


class NestedFieldEvalNode(IntrinsicNode):
    def __init__(self, fields, args, var):
        self.fields = fields
        self.args = args
        self.var = var  # the variable in which the interpolated field is written


class NestedVectorFieldNode(IntrinsicNode):
    def __getitem__(self, attr):
        return NestedVectorFieldEvalNode(self.obj, attr)


class NestedVectorFieldEvalNode(IntrinsicNode):
    def __init__(self, fields, args, var, var2, var3, var4):
        self.fields = fields
        self.args = args
        self.var = var  # the variable in which the interpolated field is written
        self.var2 = var2  # second variable for UV interpolation
        self.var3 = var3  # third variable for UVW interpolation
        self.var4 = var4  # extra variable for sigma-scaling for croco


class GridNode(IntrinsicNode):
    def __getattr__(self, attr):
        raise NotImplementedError("Access to Grids is not (yet) implemented in JIT mode")


class ConstNode(IntrinsicNode):
    def __getitem__(self, attr):
        return attr


class MathNode(IntrinsicNode):
    symbol_map = {"pi": "M_PI", "e": "M_E", "nan": "NAN"}

    def __getattr__(self, attr):
        if hasattr(math, attr):
            if attr in self.symbol_map:
                attr = self.symbol_map[attr]
            return IntrinsicNode(None, ccode=attr)
        else:
            raise AttributeError(f"Unknown math function encountered: {attr}")


class RandomNode(IntrinsicNode):
    symbol_map = {
        "random": "parcels_random",
        "uniform": "parcels_uniform",
        "randint": "parcels_randint",
        "normalvariate": "parcels_normalvariate",
        "expovariate": "parcels_expovariate",
        "vonmisesvariate": "parcels_vonmisesvariate",
        "seed": "parcels_seed",
    }

    def __getattr__(self, attr):
        if hasattr(random, attr):
            if attr in self.symbol_map:
                attr = self.symbol_map[attr]
            return IntrinsicNode(None, ccode=attr)
        else:
            raise AttributeError(f"Unknown random function encountered: {attr}")


class StatusCodeNode(IntrinsicNode):
    def __getattr__(self, attr):
        statuscodes = [c for c in vars(StatusCode) if not c.startswith("_")]
        if attr in statuscodes:
            return IntrinsicNode(None, ccode=attr.upper())
        else:
            raise AttributeError(f"Unknown status code encountered: {attr}")


class PrintNode(IntrinsicNode):
    def __init__(self):
        self.obj = "print"


class ParticleAttributeNode(IntrinsicNode):
    def __init__(self, obj, attr):
        self.ccode = f"{obj.ccode}->{attr}[pnum]"
        self.attr = attr


class ParticleXiYiZiTiAttributeNode(IntrinsicNode):
    def __init__(self, obj, attr):
        warnings.warn(
            f"Be careful when sampling particle.{attr}, as this is updated in the kernel loop. "
            "Best to place the sampling statement before advection.",
            KernelWarning,
            stacklevel=2,
        )
        self.obj = obj.ccode
        self.attr = attr


class ParticleNode(IntrinsicNode):
    def __init__(self, obj):
        super().__init__(obj, ccode="particles")

    def __getattr__(self, attr):
        if attr in ["xi", "yi", "zi", "ti"]:
            return ParticleXiYiZiTiAttributeNode(self, attr)
        if attr in [v.name for v in self.obj.variables]:
            return ParticleAttributeNode(self, attr)
        elif attr in ["delete"]:
            return ParticleAttributeNode(self, "state")
        else:
            raise AttributeError(
                f"Particle type {self.obj.name} does not define attribute '{attr}. "
                f"Please add '{attr}' as a Variable in {self.obj.name}."
            )


class IntrinsicTransformer(ast.NodeTransformer):
    """AST transformer that catches any mention of intrinsic variable
    names, such as 'particle' or 'fieldset', inserts placeholder objects
    and propagates attribute access.
    """

    def __init__(self, fieldset=None, ptype=JITParticle):
        self.fieldset = fieldset
        self.ptype = ptype

        # Counter and variable names for temporaries
        self._tmp_counter = 0
        self.tmp_vars = []
        # A stack of additional statements to be inserted
        self.stmt_stack = []

    def get_tmp(self):
        """Create a new temporary variable name."""
        tmp = "parcels_tmpvar%d" % self._tmp_counter
        self._tmp_counter += 1
        self.tmp_vars += [tmp]
        return tmp

    def visit_Name(self, node):
        """Inject IntrinsicNode objects into the tree according to keyword."""
        if node.id == "fieldset" and self.fieldset is not None:
            node = FieldSetNode(self.fieldset, ccode="fset")
        elif node.id == "particle":
            node = ParticleNode(self.ptype)
        elif node.id in ["StatusCode"]:
            node = StatusCodeNode(math, ccode="")
        elif node.id == "math":
            node = MathNode(math, ccode="")
        elif node.id in ["ParcelsRandom", "rng"]:
            node = RandomNode(math, ccode="")
        elif node.id == "print":
            node = PrintNode()
        elif (node.id == "pnum") or ("parcels_tmpvar" in node.id):
            raise NotImplementedError(f"Custom Kernels cannot contain string {node.id}; please change your kernel")
        elif node.id == "abs":
            raise NotImplementedError("abs() does not work in JIT Kernels. Use math.fabs() instead")
        return node

    def visit_Attribute(self, node):
        node.value = self.visit(node.value)
        if isinstance(node.value, IntrinsicNode):
            return getattr(node.value, node.attr)
        else:
            if node.value.id in ["np", "numpy"]:
                raise NotImplementedError(
                    "Cannot convert numpy functions in kernels to C-code.\n"
                    "Either use functions from the math library or run Parcels in Scipy mode.\n"
                    "For more information, see https://docs.oceanparcels.org/en/latest/examples/tutorial_parcels_structure.html#3.-Kernels"
                )
            elif node.value.id in ["random"]:
                raise NotImplementedError(
                    "Cannot convert random functions in kernels to C-code.\n"
                    "Use `import parcels.rng as ParcelsRandom` and then ParcelsRandom.random(), ParcelsRandom.uniform() etc.\n"
                    "For more information, see https://docs.oceanparcels.org/en/latest/examples/tutorial_parcels_structure.html#3.-Kernels"
                )
            else:
                raise NotImplementedError(f"Cannot convert '{node.value.id}' used in kernel to C-code")

    def visit_Subscript(self, node):
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)

        # If we encounter field evaluation we replace it with a
        # temporary variable and put the evaluation call on the stack.
        if isinstance(node.value, FieldNode):
            tmp = self.get_tmp()
            # Insert placeholder node for field eval ...
            self.stmt_stack += [FieldEvalNode(node.value, node.slice, tmp)]
            # .. and return the name of the temporary that will be populated
            return ast.Name(id=tmp)
        elif isinstance(node.value, VectorFieldNode):
            tmp = self.get_tmp()
            tmp2 = self.get_tmp()
            tmp3 = self.get_tmp() if "3D" in node.value.obj.vector_type else None
            tmp4 = self.get_tmp() if "3DSigma" in node.value.obj.vector_type else None
            # Insert placeholder node for field eval ...
            self.stmt_stack += [VectorFieldEvalNode(node.value, node.slice, tmp, tmp2, tmp3, tmp4)]
            # .. and return the name of the temporary that will be populated
            if tmp3:
                return ast.Tuple([ast.Name(id=tmp), ast.Name(id=tmp2), ast.Name(id=tmp3)], ast.Load())
            else:
                return ast.Tuple([ast.Name(id=tmp), ast.Name(id=tmp2)], ast.Load())
        elif isinstance(node.value, NestedFieldNode):
            tmp = self.get_tmp()
            self.stmt_stack += [NestedFieldEvalNode(node.value, node.slice, tmp)]
            return ast.Name(id=tmp)
        elif isinstance(node.value, NestedVectorFieldNode):
            tmp = self.get_tmp()
            tmp2 = self.get_tmp()
            tmp3 = self.get_tmp() if "3D" in list.__getitem__(node.value.obj, 0).vector_type else None
            tmp4 = self.get_tmp() if "3DSigma" in list.__getitem__(node.value.obj, 0).vector_type else None
            self.stmt_stack += [NestedVectorFieldEvalNode(node.value, node.slice, tmp, tmp2, tmp3, tmp4)]
            if tmp3:
                return ast.Tuple([ast.Name(id=tmp), ast.Name(id=tmp2), ast.Name(id=tmp3)], ast.Load())
            else:
                return ast.Tuple([ast.Name(id=tmp), ast.Name(id=tmp2)], ast.Load())
        else:
            return node

    def visit_AugAssign(self, node):
        node.target = self.visit(node.target)
        if isinstance(node.target, ParticleAttributeNode) and node.target.attr in ["lon", "lat", "depth", "time"]:
            warnings.warn(
                "Don't change the location of a particle directly in a Kernel. Use particle_dlon, particle_dlat, etc.",
                KernelWarning,
                stacklevel=2,
            )
        node.op = self.visit(node.op)
        node.value = self.visit(node.value)
        stmts = [node]

        # Inject statements from the stack
        if len(self.stmt_stack) > 0:
            stmts = self.stmt_stack + stmts
            self.stmt_stack = []
        return stmts

    def visit_Assign(self, node):
        node.targets = [self.visit(t) for t in node.targets]
        node.value = self.visit(node.value)
        if isinstance(node.value, ConstNode) and len(node.targets) > 0 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == node.value.ccode:
                raise NotImplementedError(
                    f"Assignment of fieldset.{node.value.ccode} to a local variable {node.targets[0].id} with same name in kernel. This is not allowed."
                )
        stmts = [node]

        # Inject statements from the stack
        if len(self.stmt_stack) > 0:
            stmts = self.stmt_stack + stmts
            self.stmt_stack = []
        return stmts

    def visit_Call(self, node):
        node.func = self.visit(node.func)
        node.args = [self.visit(a) for a in node.args]
        node.keywords = {kw.arg: self.visit(kw.value) for kw in node.keywords}

        if isinstance(node.func, ParticleAttributeNode) and node.func.attr == "state":
            node = IntrinsicNode(node, "particles->state[pnum] = DELETE")

        elif isinstance(node.func, FieldEvalCallNode):
            # get a temporary value to assign result to
            tmp = self.get_tmp()
            # whether to convert
            convert = True
            if "applyConversion" in node.keywords:
                k = node.keywords["applyConversion"]
                if isinstance(k, ast.Constant):
                    convert = k.value

            # convert args to Index(Tuple(*args))
            args = ast.Index(value=ast.Tuple(node.args, ast.Load()))

            self.stmt_stack += [FieldEvalNode(node.func.field, args, tmp, convert)]
            return ast.Name(id=tmp)

        elif isinstance(node.func, VectorFieldEvalCallNode):
            # get a temporary value to assign result to
            tmp1 = self.get_tmp()
            tmp2 = self.get_tmp()
            tmp3 = self.get_tmp() if "3D" in node.func.field.obj.vector_type else None
            tmp4 = self.get_tmp() if "3DSigma" in node.func.field.obj.vector_type else None
            # whether to convert
            convert = True
            if "applyConversion" in node.keywords:
                k = node.keywords["applyConversion"]
                if isinstance(k, ast.Constant):
                    convert = k.value

            # convert args to Index(Tuple(*args))
            args = ast.Index(value=ast.Tuple(node.args, ast.Load()))

            self.stmt_stack += [VectorFieldEvalNode(node.func.field, args, tmp1, tmp2, tmp3, tmp4, convert)]
            if tmp3:
                return ast.Tuple([ast.Name(id=tmp1), ast.Name(id=tmp2), ast.Name(id=tmp3)], ast.Load())
            else:
                return ast.Tuple([ast.Name(id=tmp1), ast.Name(id=tmp2)], ast.Load())

        return node


class TupleSplitter(ast.NodeTransformer):
    """AST transformer that detects and splits Pythonic tuple assignments into multiple statements for conversion to C."""

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            t_elts = node.targets[0].elts
            v_elts = node.value.elts
            if len(t_elts) != len(v_elts):
                raise AttributeError("Tuple lengths in assignment do not agree")
            node = [ast.Assign() for _ in t_elts]
            for n, t, v in zip(node, t_elts, v_elts, strict=True):
                n.targets = [t]
                n.value = v
        return node


class KernelGenerator(ABC, ast.NodeVisitor):
    """Code generator class that translates simple Python kernel functions into C functions.

    Works by populating and accessing the `ccode` attribute on nodes in the Python AST.
    """

    # Intrinsic variables that appear as function arguments
    kernel_vars = ["particle", "fieldset", "time", "output_time", "tol"]
    array_vars: list[str] = []

    def __init__(self, fieldset=None, ptype=JITParticle):
        self.fieldset = fieldset
        self.ptype = ptype
        self.field_args = collections.OrderedDict()
        if isinstance(fieldset.U, Field) and fieldset.U.gridindexingtype == "croco" and hasattr(fieldset, "H"):
            self.field_args["H"] = fieldset.H  # CROCO requires H field
        self.vector_field_args = collections.OrderedDict()
        self.const_args = collections.OrderedDict()

    def generate(self, py_ast, funcvars: list[str]):
        # Replace occurrences of intrinsic objects in Python AST
        transformer = IntrinsicTransformer(self.fieldset, self.ptype)
        py_ast = transformer.visit(py_ast)

        # Untangle Pythonic tuple-assignment statements
        py_ast = TupleSplitter().visit(py_ast)

        # Generate C-code for all nodes in the Python AST
        self.visit(py_ast)
        self.ccode = py_ast.ccode

        # Insert variable declarations for non-intrinsic variables
        # Make sure that repeated variables are not declared more than
        # once. If variables occur in multiple Kernels, give a warning
        used_vars: list[str] = []
        funcvars_copy = copy(funcvars)  # editing a list while looping over it is dangerous
        for kvar in funcvars:
            if kvar in used_vars + ["particle_dlon", "particle_dlat", "particle_ddepth"]:
                if kvar not in ["particle", "fieldset", "time", "particle_dlon", "particle_dlat", "particle_ddepth"]:
                    warnings.warn(
                        kvar + " declared in multiple Kernels",
                        KernelWarning,
                        stacklevel=2,
                    )
                funcvars_copy.remove(kvar)
            else:
                used_vars.append(kvar)
        funcvars = funcvars_copy
        for kvar in self.kernel_vars + self.array_vars:
            if kvar in funcvars:
                funcvars.remove(kvar)
        self.ccode.body.insert(0, c.Statement("int parcels_interp_state = 0"))
        if len(funcvars) > 0:
            for f in funcvars:
                self.ccode.body.insert(0, c.Statement(f"type_coord {f} = 0"))
        if len(transformer.tmp_vars) > 0:
            for f in transformer.tmp_vars:
                self.ccode.body.insert(0, c.Statement(f"float {f} = 0"))

        return self.ccode

    @staticmethod
    def _check_FieldSamplingArguments(ccode):
        if ccode == "particles":
            args = ("time", "particles->depth[pnum]", "particles->lat[pnum]", "particles->lon[pnum]")
        elif ccode[-1] == "particles":
            args = ccode[:-1]
        else:
            args = ccode
        return args

    def visit_FunctionDef(self, node):
        # Generate "ccode" attribute by traversing the Python AST
        for stmt in node.body:
            self.visit(stmt)

        # Create function declaration and argument list
        decl = c.Static(c.DeclSpecifier(c.Value("StatusCode", node.name), spec="inline"))
        args = [
            c.Pointer(c.Value(self.ptype.name + "p", "particles")),
            c.Value("int", "pnum"),
            c.Value("double", "time"),
        ]
        for field in self.field_args.values():
            args += [c.Pointer(c.Value("CField", f"{field.ccode_name}"))]
        for field in self.vector_field_args.values():
            for fcomponent in ["U", "V", "W"]:
                try:
                    f = getattr(field, fcomponent)
                    if f.ccode_name not in self.field_args:
                        args += [c.Pointer(c.Value("CField", f"{f.ccode_name}"))]
                        self.field_args[f.ccode_name] = f
                except:
                    pass  # field.W does not always exist
        for const, _ in self.const_args.items():
            args += [c.Value("float", const)]

        # Create function body as C-code object
        body = []
        for coord in ["lon", "lat", "depth"]:
            body += [c.Statement(f"type_coord particle_d{coord} = 0")]
            body += [c.Statement(f"particles->{coord}[pnum] = particles->{coord}_nextloop[pnum]")]
        body += [c.Statement("particles->time[pnum] = particles->time_nextloop[pnum]")]

        body += [stmt.ccode for stmt in node.body]

        for coord in ["lon", "lat", "depth"]:
            body += [c.Statement(f"particles->{coord}_nextloop[pnum] = particles->{coord}[pnum] + particle_d{coord}")]
        body += [c.Statement("particles->time_nextloop[pnum] = particles->time[pnum] + particles->dt[pnum]")]
        body += [c.Statement("return particles->state[pnum]")]
        node.ccode = c.FunctionBody(c.FunctionDeclaration(decl, args), c.Block(body))

    def visit_Call(self, node):
        """Generate C code for simple C-style function calls.

        Please note that starred and keyword arguments are currently not
        supported.
        """
        pointer_args = False
        parcels_customed_Cfunc = False
        if isinstance(node.func, PrintNode):
            # Write our own Print parser because Python3-AST does not seem to have one
            if isinstance(node.args[0], ast.Str):
                node.ccode = str(c.Statement(f'printf("{node.args[0].s}\\n")'))
            elif isinstance(node.args[0], ast.Name):
                node.ccode = str(c.Statement(f'printf("%f\\n", {node.args[0].id})'))
            elif isinstance(node.args[0], ast.BinOp):
                if hasattr(node.args[0].right, "ccode"):
                    args = node.args[0].right.ccode
                elif hasattr(node.args[0].right, "id"):
                    args = node.args[0].right.id
                elif hasattr(node.args[0].right, "elts"):
                    args = []
                    for a in node.args[0].right.elts:
                        if hasattr(a, "ccode"):
                            args.append(a.ccode)
                        elif hasattr(a, "id"):
                            args.append(a.id)
                else:
                    args = []
                s = f'printf("{node.args[0].left.s}\\n"'
                if isinstance(args, str):
                    s = s + f", {args})"
                else:
                    for arg in args:
                        s = s + (f", {arg}")
                    s = s + ")"
                node.ccode = str(c.Statement(s))
            else:
                raise RuntimeError("This print statement is not supported")
        else:
            for a in node.args:
                self.visit(a)
                if a.ccode == "parcels_customed_Cfunc_pointer_args":
                    pointer_args = True
                    parcels_customed_Cfunc = True
                elif a.ccode == "parcels_customed_Cfunc":
                    parcels_customed_Cfunc = True
                elif isinstance(a, FieldNode) or isinstance(a, VectorFieldNode):
                    a.ccode = a.obj.ccode_name
                elif isinstance(a, ParticleNode):
                    continue
                elif pointer_args:
                    a.ccode = f"&{a.ccode}"
            ccode_args = ", ".join([a.ccode for a in node.args[pointer_args:]])
            try:
                if isinstance(node.func, str):
                    node.ccode = node.func + "(" + ccode_args + ")"
                else:
                    self.visit(node.func)
                    rhs = f"{node.func.ccode}({ccode_args})"
                    if parcels_customed_Cfunc:
                        node.ccode = str(
                            c.Block(
                                [
                                    c.Assign("parcels_interp_state", rhs),
                                    c.Assign(
                                        "particles->state[pnum]", "max(particles->state[pnum], parcels_interp_state)"
                                    ),
                                    c.Statement("CHECKSTATUS_KERNELLOOP(parcels_interp_state)"),
                                ]
                            )
                        )
                    else:
                        node.ccode = rhs
            except:
                raise RuntimeError(
                    "Error in converting Kernel to C. See https://docs.oceanparcels.org/en/latest/examples/tutorial_parcels_structure.html#3.-Kernel-execution for hints and tips"
                )

    def visit_Name(self, node):
        """Catches any mention of intrinsic variable names such as 'particle' or 'fieldset' and inserts our placeholder objects."""
        if node.id == "True":
            node.id = "1"
        if node.id == "False":
            node.id = "0"
        node.ccode = node.id

    def visit_NameConstant(self, node):
        if node.value is True:
            node.ccode = "1"
        if node.value is False:
            node.ccode = "0"

    def visit_Expr(self, node):
        self.visit(node.value)
        node.ccode = c.Statement(node.value.ccode)

    def visit_Assign(self, node):
        self.visit(node.targets[0])
        self.visit(node.value)
        if isinstance(node.value, ast.List):
            # Detect in-place initialisation of multi-dimensional arrays
            tmp_node = node.value
            decl = c.Value("float", node.targets[0].id)
            while isinstance(tmp_node, ast.List):
                decl = c.ArrayOf(decl, len(tmp_node.elts))
                if isinstance(tmp_node.elts[0], ast.List):
                    # Check type and dimension are the same
                    if not all(isinstance(e, ast.List) for e in tmp_node.elts):
                        raise TypeError("Non-list element discovered in array declaration")
                    if not all(len(e.elts) == len(tmp_node.elts[0].elts) for e in tmp_node.elts):
                        raise TypeError("Irregular array length not allowed in array declaration")
                tmp_node = tmp_node.elts[0]
            node.ccode = c.Initializer(decl, node.value.ccode)
            self.array_vars += [node.targets[0].id]
        elif isinstance(node.value, ParticleXiYiZiTiAttributeNode):
            raise RuntimeError(
                f"Add index of the grid when using particle.{node.value.attr} (e.g. particle.{node.value.attr}[0])."
            )
        else:
            node.ccode = c.Assign(node.targets[0].ccode, node.value.ccode)

    def visit_AugAssign(self, node):
        self.visit(node.target)
        self.visit(node.op)
        self.visit(node.value)
        node.ccode = c.Statement(f"{node.target.ccode} {node.op.ccode}= {node.value.ccode}")

    def visit_If(self, node):
        self.visit(node.test)
        for b in node.body:
            self.visit(b)
        for b in node.orelse:
            self.visit(b)
        # field evals are replaced by a tmp variable is added to the stack.
        # Here it means field evals passes from node.test to node.body. We take it out manually
        fieldInTestCount = node.test.ccode.count("parcels_tmpvar")
        body0 = c.Block([b.ccode for b in node.body[:fieldInTestCount]])
        body = c.Block([b.ccode for b in node.body[fieldInTestCount:]])
        orelse = c.Block([b.ccode for b in node.orelse]) if len(node.orelse) > 0 else None
        ifcode = c.If(node.test.ccode, body, orelse)
        node.ccode = c.Block([body0, ifcode])

    def visit_Compare(self, node):
        self.visit(node.left)
        assert len(node.ops) == 1
        self.visit(node.ops[0])
        assert len(node.comparators) == 1
        self.visit(node.comparators[0])
        node.ccode = f"{node.left.ccode} {node.ops[0].ccode} {node.comparators[0].ccode}"

    def visit_Index(self, node):
        self.visit(node.value)
        node.ccode = node.value.ccode

    def visit_Tuple(self, node):
        for e in node.elts:
            self.visit(e)
        node.ccode = tuple([e.ccode for e in node.elts])

    def visit_List(self, node):
        for e in node.elts:
            self.visit(e)
        node.ccode = "{" + ", ".join([e.ccode for e in node.elts]) + "}"

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)
        if isinstance(node.value, FieldNode) or isinstance(node.value, VectorFieldNode):
            node.ccode = node.value.__getitem__(node.slice.ccode).ccode
        elif isinstance(node.value, ParticleXiYiZiTiAttributeNode):
            ngrid = str(self.fieldset.gridset.size if self.fieldset is not None else 1)
            node.ccode = f"{node.value.obj}->{node.value.attr}[pnum*{ngrid}+{node.slice.ccode}]"
        elif isinstance(node.value, IntrinsicNode):
            raise NotImplementedError(f"Subscript not implemented for object type {type(node.value).__name__}")
        else:
            node.ccode = f"{node.value.ccode}[{node.slice.ccode}]"

    def visit_UnaryOp(self, node):
        self.visit(node.op)
        self.visit(node.operand)
        node.ccode = f"{node.op.ccode}({node.operand.ccode})"

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.op)
        self.visit(node.right)
        if isinstance(node.op, ast.BitXor):
            raise RuntimeError(
                "JIT kernels do not support the '^' operator.\n"
                "Did you intend to use the exponential/power operator? In that case, please use '**'"
            )
        elif node.op.ccode == "pow":  # catching '**' pow statements
            node.ccode = f"pow({node.left.ccode}, {node.right.ccode})"
        else:
            node.ccode = f"({node.left.ccode} {node.op.ccode} {node.right.ccode})"
        node.s_print = True

    def visit_Add(self, node):
        node.ccode = "+"

    def visit_UAdd(self, node):
        node.ccode = "+"

    def visit_Sub(self, node):
        node.ccode = "-"

    def visit_USub(self, node):
        node.ccode = "-"

    def visit_Mult(self, node):
        node.ccode = "*"

    def visit_Div(self, node):
        node.ccode = "/"

    def visit_Mod(self, node):
        node.ccode = "%"

    def visit_Pow(self, node):
        node.ccode = "pow"

    def visit_BoolOp(self, node):
        self.visit(node.op)
        for v in node.values:
            self.visit(v)
        op_str = f" {node.op.ccode} "
        node.ccode = op_str.join([v.ccode for v in node.values])

    def visit_Eq(self, node):
        node.ccode = "=="

    def visit_NotEq(self, node):
        node.ccode = "!="

    def visit_Lt(self, node):
        node.ccode = "<"

    def visit_LtE(self, node):
        node.ccode = "<="

    def visit_Gt(self, node):
        node.ccode = ">"

    def visit_GtE(self, node):
        node.ccode = ">="

    def visit_And(self, node):
        node.ccode = "&&"

    def visit_Or(self, node):
        node.ccode = "||"

    def visit_Not(self, node):
        node.ccode = "!"

    def visit_While(self, node):
        self.visit(node.test)
        for b in node.body:
            self.visit(b)
        if len(node.orelse) > 0:
            raise RuntimeError("Else clause in while clauses cannot be translated to C")
        body = c.Block([b.ccode for b in node.body])
        node.ccode = c.DoWhile(node.test.ccode, body)

    def visit_For(self, node):
        raise RuntimeError("For loops cannot be translated to C")

    def visit_Break(self, node):
        node.ccode = c.Statement("break")

    def visit_Pass(self, node):
        node.ccode = c.Statement("")

    def visit_FieldNode(self, node):
        """Record intrinsic fields used in kernel."""
        self.field_args[node.obj.ccode_name] = node.obj

    def visit_NestedFieldNode(self, node):
        """Record intrinsic fields used in kernel."""
        for fld in node.obj:
            self.field_args[fld.ccode_name] = fld

    def visit_VectorFieldNode(self, node):
        """Record intrinsic fields used in kernel."""
        self.vector_field_args[node.obj.ccode_name] = node.obj

    def visit_NestedVectorFieldNode(self, node):
        """Record intrinsic fields used in kernel."""
        for fld in node.obj:
            self.vector_field_args[fld.ccode_name] = fld

    def visit_ConstNode(self, node):
        self.const_args[node.ccode] = node.obj

    def visit_Return(self, node):
        self.visit(node.value)
        node.ccode = c.Statement(f"return {node.value.ccode}")

    def visit_FieldEvalNode(self, node):
        self.visit(node.field)
        self.visit(node.args)
        args = self._check_FieldSamplingArguments(node.args.ccode)
        statements_croco = []
        if "croco" in node.field.obj.gridindexingtype and node.field.obj.name != "H":  # TODO needs to be sigma
            statements_croco.append(
                c.Assign(
                    "parcels_interp_state",
                    f"temporal_interpolation({args[3]}, {args[2]}, 0, time, H, &particles->xi[pnum*ngrid], &particles->yi[pnum*ngrid], &particles->zi[pnum*ngrid], &particles->ti[pnum*ngrid], &{node.var}, LINEAR, {node.field.obj.gridindexingtype.upper()})",
                )
            )
            statements_croco.append(c.Statement(f"{node.var} = {args[1]}/{node.var}"))
            args = (args[0], node.var, args[2], args[3])
        ccode_eval = node.field.obj.ccode_eval(node.var, *args)
        stmts = [
            c.Assign("parcels_interp_state", ccode_eval),
            c.Assign("particles->state[pnum]", "max(particles->state[pnum], parcels_interp_state)"),
        ]

        if node.convert:
            ccode_conv = node.field.obj.ccode_convert(*args)
            conv_stat = c.Statement(f"{node.var} *= {ccode_conv}")
            stmts += [conv_stat]

        node.ccode = c.Block(statements_croco + stmts + [c.Statement("CHECKSTATUS_KERNELLOOP(parcels_interp_state)")])

    def visit_VectorFieldEvalNode(self, node):
        self.visit(node.field)
        self.visit(node.args)
        args = self._check_FieldSamplingArguments(node.args.ccode)
        statements_croco = []
        if "3DSigma" in node.field.obj.vector_type:
            statements_croco.append(
                c.Assign(
                    "parcels_interp_state",
                    f"temporal_interpolation({args[3]}, {args[2]}, 0, time, H, &particles->xi[pnum*ngrid], &particles->yi[pnum*ngrid], &particles->zi[pnum*ngrid], &particles->ti[pnum*ngrid], &{node.var}, LINEAR, {node.field.obj.U.gridindexingtype.upper()})",
                )
            )
            statements_croco.append(c.Statement(f"{node.var4} = {args[1]}/{node.var}"))
            args = (args[0], node.var4, args[2], args[3])
        ccode_eval = node.field.obj.ccode_eval(
            node.var, node.var2, node.var3, node.field.obj.U, node.field.obj.V, node.field.obj.W, *args
        )
        if node.convert and node.field.obj.U.interp_method != "cgrid_velocity":
            ccode_conv1 = node.field.obj.U.ccode_convert(*args)
            ccode_conv2 = node.field.obj.V.ccode_convert(*args)
            statements = [c.Statement(f"{node.var} *= {ccode_conv1}"), c.Statement(f"{node.var2} *= {ccode_conv2}")]
        else:
            statements = []
        if node.convert and "3D" in node.field.obj.vector_type:
            ccode_conv3 = node.field.obj.W.ccode_convert(*args)
            statements.append(c.Statement(f"{node.var3} *= {ccode_conv3}"))
        conv_stat = c.Block(statements)
        node.ccode = c.Block(
            [
                c.Block(statements_croco),
                c.Assign("parcels_interp_state", ccode_eval),
                c.Assign("particles->state[pnum]", "max(particles->state[pnum], parcels_interp_state)"),
                conv_stat,
                c.Statement("CHECKSTATUS_KERNELLOOP(parcels_interp_state)"),
            ]
        )

    def visit_NestedFieldEvalNode(self, node):
        self.visit(node.fields)
        self.visit(node.args)
        cstat = []
        args = self._check_FieldSamplingArguments(node.args.ccode)
        for fld in node.fields.obj:
            ccode_eval = fld.ccode_eval(node.var, *args)
            ccode_conv = fld.ccode_convert(*args)
            conv_stat = c.Statement(f"{node.var} *= {ccode_conv}")
            cstat += [
                c.Assign("particles->state[pnum]", ccode_eval),
                conv_stat,
                c.If(
                    "particles->state[pnum] != ERROROUTOFBOUNDS ",
                    c.Block([c.Statement("CHECKSTATUS_KERNELLOOP(particles->state[pnum])"), c.Statement("break")]),
                ),
            ]
        cstat += [c.Statement("CHECKSTATUS_KERNELLOOP(particles->state[pnum])"), c.Statement("break")]
        node.ccode = c.While("1==1", c.Block(cstat))

    def visit_NestedVectorFieldEvalNode(self, node):
        self.visit(node.fields)
        self.visit(node.args)
        cstat = []
        args = self._check_FieldSamplingArguments(node.args.ccode)
        for fld in node.fields.obj:
            ccode_eval = fld.ccode_eval(node.var, node.var2, node.var3, fld.U, fld.V, fld.W, *args)
            if fld.U.interp_method != "cgrid_velocity":
                ccode_conv1 = fld.U.ccode_convert(*args)
                ccode_conv2 = fld.V.ccode_convert(*args)
                statements = [c.Statement(f"{node.var} *= {ccode_conv1}"), c.Statement(f"{node.var2} *= {ccode_conv2}")]
            else:
                statements = []
            if "3D" in fld.vector_type:
                ccode_conv3 = fld.W.ccode_convert(*args)
                statements.append(c.Statement(f"{node.var3} *= {ccode_conv3}"))
            cstat += [
                c.Assign("particles->state[pnum]", ccode_eval),
                c.Block(statements),
                c.If(
                    "particles->state[pnum] != ERROROUTOFBOUNDS ",
                    c.Block([c.Statement("CHECKSTATUS_KERNELLOOP(particles->state[pnum])"), c.Statement("break")]),
                ),
            ]
        cstat += [c.Statement("CHECKSTATUS_KERNELLOOP(particles->state[pnum])"), c.Statement("break")]
        node.ccode = c.While("1==1", c.Block(cstat))

    def visit_Print(self, node):
        for n in node.values:
            self.visit(n)
        if hasattr(node.values[0], "s"):
            node.ccode = c.Statement(f'printf("{n.ccode}\\n")')
            return
        if hasattr(node.values[0], "s_print"):
            args = node.values[0].right.ccode
            s = f'printf("{node.values[0].left.ccode}\\n"'
            if isinstance(args, str):
                s = s + f", {args})"
            else:
                for arg in args:
                    s = s + (f", {arg}")
                s = s + ")"
            node.ccode = c.Statement(s)
            return
        vars = ", ".join([n.ccode for n in node.values])
        int_vars = ["particle->id", "particle->xi", "particle->yi", "particle->zi"]
        stat = ", ".join(["%d" if n.ccode in int_vars else "%f" for n in node.values])
        node.ccode = c.Statement(f'printf("{stat}\\n", {vars})')

    def visit_Constant(self, node):
        if node.value == "parcels_customed_Cfunc_pointer_args":
            node.ccode = node.value
        elif isinstance(node.value, str):
            node.ccode = ""  # skip strings from docstrings or comments
        elif isinstance(node.value, bool):
            node.ccode = "1" if node.value is True else "0"
        else:
            node.ccode = str(node.value)


class LoopGenerator:
    """Code generator class that adds type definitions and the outer loop around kernel functions to generate compilable C code."""

    def __init__(self, fieldset, ptype=None):
        self.fieldset = fieldset
        self.ptype = ptype

    def generate(self, funcname, field_args, const_args, kernel_ast, c_include):
        ccode = []

        pname = self.ptype.name + "p"

        # ==== Add include for Parcels and math header ==== #
        ccode += [str(c.Include("parcels.h", system=False))]
        ccode += [str(c.Include("math.h", system=False))]
        ccode += [str(c.Assign("const int ngrid", str(self.fieldset.gridset.size if self.fieldset is not None else 1)))]

        # ==== Generate type definition for particle type ==== #
        vdeclp = [c.Pointer(c.POD(v.dtype, v.name)) for v in self.ptype.variables]
        ccode += [str(c.Typedef(c.GenerableStruct("", vdeclp, declname=pname)))]

        if c_include:
            ccode += [c_include]

        # ==== Insert kernel code ==== #
        ccode += [str(kernel_ast)]

        # Generate outer loop for repeated kernel invocation
        args = [
            c.Value("int", "num_particles"),
            c.Pointer(c.Value(pname, "particles")),
            c.Value("double", "endtime"),
            c.Value("double", "dt"),
        ]
        for field, _ in field_args.items():
            args += [c.Pointer(c.Value("CField", f"{field}"))]
        for const, _ in const_args.items():
            args += [c.Value("double", const)]  # are we SURE those const's are double's ?
        fargs_str = ", ".join(["particles->time_nextloop[pnum]"] + list(field_args.keys()) + list(const_args.keys()))
        # ==== statement clusters use to compose 'body' variable and variables 'time_loop' and 'part_loop' ==== ##
        sign_dt = c.Assign("sign_dt", "dt > 0 ? 1 : -1")

        # ==== check if next_dt is in the particle type ==== #
        dtname = "next_dt" if "next_dt" in [v.name for v in self.ptype.variables] else "dt"

        # ==== main computation body ==== #
        body = []
        body += [c.Value("double", "pre_dt")]
        body += [c.Statement("pre_dt = particles->dt[pnum]")]
        body += [c.If("sign_dt*particles->time_nextloop[pnum] >= sign_dt*(endtime)", c.Statement("break"))]
        body += [
            c.If(
                f"fabs(endtime - particles->time_nextloop[pnum]) < fabs(particles->{dtname}[pnum])-1e-6",
                c.Statement(f"particles->{dtname}[pnum] = fabs(endtime - particles->time_nextloop[pnum]) * sign_dt"),
            )
        ]
        body += [c.Assign("particles->state[pnum]", f"{funcname}(particles, pnum, {fargs_str})")]
        body += [
            c.If(
                "particles->state[pnum] == SUCCESS",
                c.Block(
                    [
                        c.If(
                            "sign_dt*particles->time[pnum] < sign_dt*endtime",
                            c.Block([c.Assign("particles->state[pnum]", "EVALUATE")]),
                            c.Block([c.Assign("particles->state[pnum]", "SUCCESS")]),
                        )
                    ]
                ),
            )
        ]
        body += [c.If("particles->state[pnum] == STOPALLEXECUTION", c.Statement("return"))]
        body += [c.Statement("particles->dt[pnum] = pre_dt")]
        body += [
            c.If(
                "(particles->state[pnum] == REPEAT || particles->state[pnum] == DELETE)",
                c.Block([c.Statement("break")]),
            )
        ]

        time_loop = c.While("(particles->state[pnum] == EVALUATE || particles->state[pnum] == REPEAT)", c.Block(body))
        part_loop = c.For("pnum = 0", "pnum < num_particles", "++pnum", c.Block([time_loop]))
        fbody = c.Block(
            [
                c.Value("int", "pnum"),
                c.Value("double", "sign_dt"),
                sign_dt,
                part_loop,
            ]
        )
        fdecl = c.FunctionDeclaration(c.Value("void", "particle_loop"), args)
        ccode += [str(c.FunctionBody(fdecl, fbody))]
        return "\n\n".join(ccode)
