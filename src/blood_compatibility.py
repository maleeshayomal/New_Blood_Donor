"""
Blood Group Compatibility Module (Rule-Based Deductive Reasoning)
==================================================================
Academic AI Decision-Support System

DISCLAIMER:
This module contains standard simplified red blood cell (RBC) compatibility rules
for academic demonstration and decision support only. Clinical transfusion decisions,
minor antigen matching, cross-matching, and donor medical eligibility must always be
verified by certified medical professionals.
"""

from typing import List, Dict

# Complete standard red blood cell (RBC) compatibility matrix:
# Mapping: Recipient Blood Group -> List of Compatible Donor Blood Groups
RECIPIENT_TO_DONOR_COMPATIBILITY: Dict[str, List[str]] = {
    "O-": ["O-"],
    "O+": ["O-", "O+"],
    "A-": ["O-", "A-"],
    "A+": ["O-", "O+", "A-", "A+"],
    "B-": ["O-", "B-"],
    "B+": ["O-", "O+", "B-", "B+"],
    "AB-": ["O-", "A-", "B-", "AB-"],
    "AB+": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
}

# Reverse mapping: Donor Blood Group -> List of Compatible Recipient Blood Groups
DONOR_TO_RECIPIENT_COMPATIBILITY: Dict[str, List[str]] = {
    "O-": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],  # Universal Red Cell Donor
    "O+": ["O+", "A+", "B+", "AB+"],
    "A-": ["A-", "A+", "AB-", "AB+"],
    "A+": ["A+", "AB+"],
    "B-": ["B-", "B+", "AB-", "AB+"],
    "B+": ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"]  # Universal Red Cell Recipient
}

ALL_BLOOD_GROUPS: List[str] = ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]


def is_compatible(donor_group: str, recipient_group: str) -> bool:
    """
    Check whether a given donor blood group is compatible with a recipient blood group.

    Parameters
    ----------
    donor_group : str
        Blood group of the potential donor (e.g., 'O+', 'AB-')
    recipient_group : str
        Blood group of the emergency recipient (e.g., 'A+')

    Returns
    -------
    bool
        True if the donor group is compatible with the recipient, False otherwise.
    """
    donor_clean = str(donor_group).strip().upper()
    recipient_clean = str(recipient_group).strip().upper()
    
    compatible_donors = RECIPIENT_TO_DONOR_COMPATIBILITY.get(recipient_clean, [])
    return donor_clean in compatible_donors


def get_compatible_donors(recipient_group: str) -> List[str]:
    """
    Retrieve the list of donor blood groups compatible with the given recipient group.

    Parameters
    ----------
    recipient_group : str
        Recipient's required blood group.

    Returns
    -------
    List[str]
        List of compatible donor blood groups.
    """
    recipient_clean = str(recipient_group).strip().upper()
    return RECIPIENT_TO_DONOR_COMPATIBILITY.get(recipient_clean, [])


def get_compatible_recipients(donor_group: str) -> List[str]:
    """
    Retrieve the list of recipient blood groups that can receive from the given donor group.

    Parameters
    ----------
    donor_group : str
        Donor's blood group.

    Returns
    -------
    List[str]
        List of compatible recipient blood groups.
    """
    donor_clean = str(donor_group).strip().upper()
    return DONOR_TO_RECIPIENT_COMPATIBILITY.get(donor_clean, [])


def get_compatibility_description(donor_group: str, recipient_group: str) -> str:
    """
    Generate an explainable textual description of the compatibility relationship.
    """
    donor_clean = str(donor_group).strip().upper()
    recipient_clean = str(recipient_group).strip().upper()
    
    if is_compatible(donor_clean, recipient_clean):
        if donor_clean == recipient_clean:
            return f"Identical match ({donor_clean} to {recipient_clean})"
        elif donor_clean == "O-":
            return f"Universal donor match (O- to {recipient_clean})"
        elif recipient_clean == "AB+":
            return f"Universal recipient match ({donor_clean} to AB+)"
        else:
            return f"Compatible cross-group match ({donor_clean} to {recipient_clean})"
    else:
        return f"Incompatible blood group match ({donor_clean} cannot donate to {recipient_clean})"
