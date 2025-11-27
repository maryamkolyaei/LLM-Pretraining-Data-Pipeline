def install_packages():
    import subprocess
    """
    Installs the required Python packages using pip.
    """
    packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "langdetect",
        "fastparquet",
        "lxml",
        "transformers"
    ]

    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run(
                ["pip", "install", "-U", package],
                check=True,  # Raises CalledProcessError if the command fails
                text=True    # Ensures output is captured as a string
            )
            print(f"{package} installed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}\n")

if __name__ == "__main__":
    install_packages()