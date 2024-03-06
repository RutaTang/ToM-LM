# Use the latest Ubuntu image
FROM haskell:8.10.7
LABEL author="Weizhi Tang"

# Set the working directory
WORKDIR /app

# Install stack
RUN curl -sSL https://get.haskellstack.org/ | sh
# Verify installations
RUN stack --version

# Copy the project into the container
COPY . /app

# Build your SMCDEL
WORKDIR /app/executors/SMCDEL
RUN stack build
RUN stack install

# Entrypoint
WORKDIR /app
ENTRYPOINT ["smcdel", "MuddyShort.smcdel.txt"]