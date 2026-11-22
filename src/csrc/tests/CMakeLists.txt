include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.16.0
)

FetchContent_MakeAvailable(googletest)

add_executable(unit_tests
  test_mesh.cpp
  # ...
)

target_link_libraries(unit_tests
  PRIVATE
  numerical_pdes
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)


# R Run test
# ctest

# # Run all tests with verbose output
# ctest --verbose

# # Run tests in parallel
# ctest -j$(nproc) --output-on-failure

# # Run tests matching a name pattern
# ctest -R <Pattern> --output-on-failure

# # List discovered tests without running them
# ctest -N