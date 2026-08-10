"""Runtime-generated protobuf descriptors for the checked-in Aether schema.

The project keeps the schema in ``proto/aether.proto``.  This binding is
checked into the source distribution so a developer does not need protoc or
grpcio-tools installed merely to import the client.  The descriptor is built
with the same protobuf reflection APIs used by protoc-generated modules and
therefore exposes normal typed protobuf message classes.
"""

from __future__ import annotations

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, struct_pb2  # noqa: F401


def _optional_scalar(
    message: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    field_type: int,
) -> None:
    oneof = message.oneof_decl.add()
    oneof.name = f"_{name}"
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = field_type
    field.proto3_optional = True
    field.oneof_index = len(message.oneof_decl) - 1


_file = descriptor_pb2.FileDescriptorProto()
_file.name = "aether.proto"
_file.package = "aether"
_file.syntax = "proto3"
_file.dependency.append("google/protobuf/struct.proto")

request = _file.message_type.add()
request.name = "GenerateRequest"
for field_name, field_number, field_type in (
    ("max_tokens", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ("temperature", 4, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
    ("top_p", 5, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
    ("top_k", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
):
    _optional_scalar(request, field_name, field_number, field_type)
for field_name, field_number in (("model_id", 1), ("prompt", 2)):
    field = request.field.add()
    field.name = field_name
    field.number = field_number
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
stop = request.field.add()
stop.name = "stop"
stop.number = 7
stop.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
stop.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

response = _file.message_type.add()
response.name = "GenerateResponse"
for field_name, field_number, field_type in (
    ("text", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("prompt_tokens", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ("completion_tokens", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ("total_tokens", 4, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ("finish_reason", 5, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("backend_name", 6, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
):
    field = response.field.add()
    field.name = field_name
    field.number = field_number
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = field_type
metrics = response.field.add()
metrics.name = "metrics"
metrics.number = 7
metrics.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
metrics.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
metrics.type_name = ".google.protobuf.Struct"

chunk = _file.message_type.add()
chunk.name = "GenerateChunk"
for field_name, field_number, field_type in (
    ("text", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
    ("index", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ("final", 3, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
):
    field = chunk.field.add()
    field.name = field_name
    field.number = field_number
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = field_type

health_request = _file.message_type.add()
health_request.name = "HealthRequest"
health_response = _file.message_type.add()
health_response.name = "HealthResponse"
for field_name, field_number in (("status", 1), ("service", 2)):
    field = health_response.field.add()
    field.name = field_name
    field.number = field_number
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

service = _file.service.add()
service.name = "AetherRuntime"
for method_name, input_name, output_name, server_streaming in (
    ("Generate", "GenerateRequest", "GenerateResponse", False),
    ("GenerateStream", "GenerateRequest", "GenerateChunk", True),
    ("Health", "HealthRequest", "HealthResponse", False),
):
    method = service.method.add()
    method.name = method_name
    method.input_type = f".aether.{input_name}"
    method.output_type = f".aether.{output_name}"
    method.server_streaming = server_streaming

DESCRIPTOR = descriptor_pool.Default().AddSerializedFile(_file.SerializeToString())
GenerateRequest = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["GenerateRequest"])
GenerateResponse = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["GenerateResponse"])
GenerateChunk = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["GenerateChunk"])
HealthRequest = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["HealthRequest"])
HealthResponse = message_factory.GetMessageClass(DESCRIPTOR.message_types_by_name["HealthResponse"])

__all__ = [
    "DESCRIPTOR",
    "GenerateChunk",
    "GenerateRequest",
    "GenerateResponse",
    "HealthRequest",
    "HealthResponse",
]
