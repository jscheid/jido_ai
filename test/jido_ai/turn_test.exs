defmodule Jido.AI.TurnTest do
  use ExUnit.Case, async: true

  alias Jido.AI.Turn

  @moduletag :unit

  defmodule Calculator do
    use Jido.Action,
      name: "calculator",
      description: "Test calculator",
      schema:
        Zoi.object(%{
          operation: Zoi.string(),
          a: Zoi.integer(),
          b: Zoi.integer()
        })

    def run(%{operation: "add", a: a, b: b}, _context), do: {:ok, %{result: a + b}}
    def run(_params, _context), do: {:error, :unsupported_operation}
  end

  describe "from_response/2" do
    test "classifies final answer and extracts thinking/text content" do
      response = %{
        message: %{
          content: [
            %{type: :thinking, thinking: "let me think"},
            %{type: :text, text: "hello"},
            %{type: :text, text: "world"}
          ],
          metadata: %{response_id: "resp_plain_1"},
          tool_calls: nil
        },
        finish_reason: :stop,
        usage: %{input_tokens: 10, output_tokens: 5}
      }

      turn = Turn.from_response(response, model: "anthropic:claude-haiku-4-5")

      assert turn.type == :final_answer
      assert turn.text == "hello\nworld"
      assert turn.thinking_content == "let me think"
      assert turn.tool_calls == []
      assert turn.usage == %{input_tokens: 10, output_tokens: 5}
      assert turn.model == "anthropic:claude-haiku-4-5"
      assert turn.message_metadata == %{response_id: "resp_plain_1"}
    end

    test "classifies tool calls from finish reason and tool call payload" do
      response = %{
        message: %{
          content: "",
          tool_calls: [
            %{id: "tc_1", name: "calculator", arguments: %{a: 1, b: 2}}
          ]
        },
        finish_reason: :tool_calls,
        usage: %{"input_tokens" => "2", "output_tokens" => "3"}
      }

      turn = Turn.from_response(response)

      assert turn.type == :tool_calls
      assert length(turn.tool_calls) == 1
      assert turn.usage == %{input_tokens: 2, output_tokens: 3}
      assert Turn.needs_tools?(turn)
    end

    test "uses ReqLLM.Response classification for canonical responses" do
      reasoning_details = [
        %ReqLLM.Message.ReasoningDetails{
          text: "Need a tool",
          signature: "sig_1",
          encrypted?: false,
          provider: :anthropic,
          format: "thinking/v1",
          index: 0,
          provider_data: %{}
        }
      ]

      response = %ReqLLM.Response{
        id: "resp_1",
        model: "anthropic:claude-haiku-4-5",
        context: ReqLLM.Context.new(),
        message:
          ReqLLM.Context.assistant("",
            tool_calls: [ReqLLM.ToolCall.new("tc_1", "calculator", ~s({"a":1,"b":2}))],
            metadata: %{response_id: "resp_1"}
          )
          |> Map.put(:reasoning_details, reasoning_details),
        stream?: false,
        stream: nil,
        usage: %{"input_tokens" => "4", "output_tokens" => "2"},
        finish_reason: :tool_calls,
        provider_meta: %{},
        error: nil
      }

      turn = Turn.from_response(response)

      assert turn.type == :tool_calls
      assert turn.text == ""
      assert turn.thinking_content == nil
      assert turn.reasoning_details == reasoning_details
      assert turn.tool_calls == [%{id: "tc_1", name: "calculator", arguments: %{"a" => 1, "b" => 2}}]
      assert turn.usage == %{input_tokens: 4, output_tokens: 2}
      assert turn.model == "anthropic:claude-haiku-4-5"
      assert turn.message_metadata == %{response_id: "resp_1"}
    end
  end

  describe "message projections" do
    test "projects assistant message metadata, reasoning_details, and tool messages as ReqLLM messages" do
      reasoning_details = [%{signature: "sig_123"}]

      turn =
        %Turn{
          type: :tool_calls,
          text: "",
          tool_calls: [%{id: "tc_1", name: "calculator", arguments: %{a: 5, b: 3}}],
          message_metadata: %{response_id: "resp_tool_round_1"},
          reasoning_details: reasoning_details
        }
        |> Turn.with_tool_results([
          %{id: "tc_1", name: "calculator", content: "{\"result\":8}", raw_result: {:ok, %{result: 8}, []}}
        ])

      assistant_message = Turn.assistant_message(turn)
      [tool_message] = Turn.tool_messages(turn)

      assert %ReqLLM.Message{} = assistant_message
      assert assistant_message.role == :assistant
      assert assistant_message.metadata == %{response_id: "resp_tool_round_1"}
      assert assistant_message.reasoning_details == reasoning_details
      assert [%ReqLLM.ToolCall{} = tool_call] = assistant_message.tool_calls
      assert tool_call.id == "tc_1"
      assert ReqLLM.ToolCall.name(tool_call) == "calculator"
      assert ReqLLM.ToolCall.args_map(tool_call) == %{"a" => 5, "b" => 3}

      assert %ReqLLM.Message{} = tool_message
      assert tool_message.role == :tool
      assert tool_message.tool_call_id == "tc_1"
      assert tool_message.name == "calculator"
      assert Turn.extract_from_content(tool_message.content) == "{\"result\":8}"
    end
  end

  describe "format_tool_result_content/1" do
    test "formats common success and error shapes" do
      assert Turn.format_tool_result_content({:ok, %{value: 1}}) == "{\"value\":1}"
      assert Turn.format_tool_result_content({:error, %{message: "boom"}}) == "boom"
      assert Turn.format_tool_result_content({:error, :badarg}) == ":badarg"
    end
  end

  describe "format_tool_result_content with %Result{}" do
    alias Jido.Action.{Result, ContentPart}

    test "returns content parts when Result has non-empty content" do
      result =
        {:ok,
         %Result{
           data: %{exit_code: 0},
           content: [ContentPart.text("Image loaded"), ContentPart.image(<<1, 2>>, "image/png")]
         }}

      content = Turn.format_tool_result_content(result)
      assert is_list(content)
      assert length(content) == 2

      assert [
               %ReqLLM.Message.ContentPart{type: :text, text: "Image loaded"},
               %ReqLLM.Message.ContentPart{type: :image, data: <<1, 2>>, media_type: "image/png"}
             ] = content
    end

    test "falls back to JSON-encoded data when content is empty" do
      result = {:ok, %Result{data: %{exit_code: 0, stdout: "ok"}, content: []}}
      content = Turn.format_tool_result_content(result)
      assert is_binary(content)
      assert content =~ "exit_code"
    end
  end

  describe "format_tool_result_content with 3-tuple %Result{}" do
    alias Jido.Action.{Result, ContentPart}

    test "3-tuple {:ok, %Result{content: [...]}, effects} returns content parts" do
      result = {:ok, %Result{data: %{x: 1}, content: [ContentPart.text("hi")]}, [:eff]}
      content = Turn.format_tool_result_content(result)
      assert [%ReqLLM.Message.ContentPart{type: :text, text: "hi"}] = content
    end

    test "3-tuple {:ok, %Result{content: []}, effects} falls back to data" do
      result = {:ok, %Result{data: %{x: 1}, content: []}, [:eff]}
      content = Turn.format_tool_result_content(result)
      assert is_binary(content)
      assert content =~ "x"
    end
  end

  describe "run_tools/3 with %Result{}" do
    alias Jido.Action.{Result, ContentPart}

    defmodule ImageAction do
      use Jido.Action,
        name: "image_viewer",
        description: "Returns an image via Result",
        schema: Zoi.object(%{path: Zoi.string()})

      def run(_params, _context) do
        {:ok,
         %Result{
           data: %{stdout: "Image loaded"},
           content: [ContentPart.image(<<0xFF, 0xD8>>, "image/jpeg")]
         }}
      end
    end

    @tag capture_log: true
    test "action returning %Result{} with content produces multimodal tool result" do
      turn = %Turn{
        type: :tool_calls,
        text: "",
        tool_calls: [
          %{id: "tc_img", name: "image_viewer", arguments: %{"path" => "/test.jpg"}}
        ]
      }

      context = %{tools: %{ImageAction.name() => ImageAction}}

      assert {:ok, updated_turn} = Turn.run_tools(turn, context, timeout: 5000)
      assert length(updated_turn.tool_results) == 1

      [tool_result] = updated_turn.tool_results
      assert tool_result.id == "tc_img"
      assert tool_result.name == "image_viewer"

      # Content should be a list of ReqLLM ContentParts, not a string
      assert is_list(tool_result.content), "Expected list of ContentParts, got: #{inspect(tool_result.content)}"

      assert [%ReqLLM.Message.ContentPart{type: :image, data: <<0xFF, 0xD8>>, media_type: "image/jpeg"}] =
               tool_result.content
    end
  end

  describe "run_tools/3" do
    test "executes tool calls and appends normalized tool results" do
      turn = %Turn{
        type: :tool_calls,
        text: "",
        tool_calls: [
          %{id: "tc_1", name: "calculator", arguments: %{"operation" => "add", "a" => 5, "b" => 3}}
        ]
      }

      context = %{tools: %{Calculator.name() => Calculator}}

      assert {:ok, updated_turn} = Turn.run_tools(turn, context, timeout: 1000)
      assert length(updated_turn.tool_results) == 1

      [tool_result] = updated_turn.tool_results
      assert tool_result.id == "tc_1"
      assert tool_result.name == "calculator"
      assert tool_result.content == "{\"result\":8}"
      assert tool_result.raw_result == {:ok, %{result: 8}, []}
    end

    test "returns original turn when no tool calls are requested" do
      turn = %Turn{type: :final_answer, text: "done", tool_calls: []}
      assert {:ok, ^turn} = Turn.run_tools(turn, %{})
    end
  end

  describe "tool execution telemetry" do
    test "execute_module emits duration_ms measurement on stop events" do
      test_pid = self()
      handler_id = "turn-stop-#{System.unique_integer([:positive])}"

      :ok =
        :telemetry.attach(
          handler_id,
          [:jido, :ai, :tool, :execute, :stop],
          fn _event, measurements, _metadata, _config ->
            send(test_pid, {:stop_measurements, measurements})
          end,
          nil
        )

      on_exit(fn -> :telemetry.detach(handler_id) end)

      assert {:ok, _result, _effects} =
               Turn.execute_module(
                 Calculator,
                 %{operation: "add", a: 1, b: 2},
                 %{observability: %{emit_telemetry?: true}}
               )

      assert_receive {:stop_measurements, measurements}
      assert is_integer(measurements.duration_ms)
      assert measurements.duration_ms >= 0
      assert is_integer(measurements.duration)
    end
  end
end
