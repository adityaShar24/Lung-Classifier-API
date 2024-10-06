from enum import Enum


class ChoiceEnum(Enum):

    @classmethod
    def choices(cls):
        return tuple((i.value, i.name) for i in cls)

    @classmethod
    def help(cls):
        return ", ".join([f"{i.name} -> {i.value}" for i in cls])

    @classmethod
    def values(cls):
        return [i.value for i in cls]


class GenderEnum(ChoiceEnum):
    Male = "Male"
    Female = "Female"
    Other = "Other"


class RoleEnum(ChoiceEnum):
    Doctor = 'Doctor'
    Patient = 'Patient'