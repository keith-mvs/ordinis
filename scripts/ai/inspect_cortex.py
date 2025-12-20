# inspect_cortex.py
import inspect

from ordinis.engines.cortex.core.engine import CortexEngine


def main():
    # Print the full signature of the constructor
    sig = inspect.signature(CortexEngine.__init__)
    print("CortexEngine.__init__ signature:")
    print(sig)

    # (Optional) Show the docstring for extra clues
    print("\nDocstring:")
    print(inspect.getdoc(CortexEngine.__init__) or "No docstring available")


if __name__ == "__main__":
    main()
